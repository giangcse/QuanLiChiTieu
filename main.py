# -*- coding: utf-8 -*-

import logging
import re
import pandas as pd
from datetime import datetime
from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Thư viện cho Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# Thư viện để làm việc với file .env
import os
from dotenv import load_dotenv

# Thư viện để vẽ biểu đồ
import matplotlib.pyplot as plt
import io

# --- CẤU HÌNH ---
# Tải các biến môi trường từ file .env vào hệ thống
load_dotenv()

# Lấy token từ biến môi trường đã tải
TOKEN = os.getenv('TELEGRAM_API_TOKEN')
DATA_FILE = 'thu_chi_data.csv'

# Bật logging để theo dõi và gỡ lỗi
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- BỘ PHẬN MÁY HỌC ---
expense_model = None
income_model = None

def train_models():
    """Huấn luyện đồng thời cả hai mô hình phân loại cho Thu và Chi."""
    global expense_model, income_model
    logger.info("Bắt đầu huấn luyện các mô hình với dữ liệu mở rộng...")
    
    # --- Dữ liệu huấn luyện cho CHI TIÊU (Mở rộng) ---
    expense_data = [
        # Ăn uống
        ("cơm sườn trưa", "Ăn uống"), ("mua ly trà sữa", "Ăn uống"), ("cà phê với bạn", "Ăn uống"),
        ("ăn tối nhà hàng", "Ăn uống"), ("bún chả", "Ăn uống"), ("phở bò", "Ăn uống"),
        ("đi ăn lẩu", "Ăn uống"), ("mua đồ ăn vặt", "Ăn uống"), ("thanh toán ahamove", "Ăn uống"),
        ("trà đá", "Ăn uống"), ("đi chợ mua thức ăn", "Ăn uống"),
        # Đi lại
        ("đổ xăng xe máy", "Đi lại"), ("vé xe bus tháng", "Đi lại"), ("tiền grab đi làm", "Đi lại"),
        ("gửi xe", "Đi lại"), ("tiền vé máy bay", "Đi lại"), ("phí cầu đường", "Đi lại"),
        ("bảo dưỡng xe", "Đi lại"), ("rửa xe", "Đi lại"),
        # Mua sắm
        ("mua áo sơ mi", "Mua sắm"), ("đặt hàng shopee", "Mua sắm"), ("mua một đôi giày mới", "Mua sắm"),
        ("mua sách", "Mua sắm"), ("mua đồ gia dụng", "Mua sắm"), ("mua quà sinh nhật", "Mua sắm"),
        ("thanh toán tiki", "Mua sắm"), ("mua sắm lazada", "Mua sắm"),
        # Hóa đơn
        ("thanh toán tiền điện", "Hóa đơn"), ("đóng tiền net FPT", "Hóa đơn"), ("tiền nhà tháng 8", "Hóa đơn"),
        ("tiền mạng viettel", "Hóa đơn"), ("phí chung cư", "Hóa đơn"), ("truyền hình cáp", "Hóa đơn"),
        ("nạp tiền điện thoại", "Hóa đơn"),
        # Giải trí
        ("vé xem phim cgv", "Giải trí"), ("mua vé concert", "Giải trí"), ("đi bar", "Giải trí"),
        ("đăng ký gym", "Giải trí"), ("mua game trên steam", "Giải trí"),
        # Sức khỏe
        ("mua thuốc cảm", "Sức khỏe"), ("tiền khám răng", "Sức khỏe"), ("mua vitamin", "Sức khỏe"),
        ("khám bệnh", "Sức khỏe"),
        # Giáo dục
        ("học phí khóa học online", "Giáo dục"), ("mua tài liệu học", "Giáo dục"), ("đóng tiền học", "Giáo dục"),
    ]
    expense_descriptions = [item[0] for item in expense_data]
    expense_categories = [item[1] for item in expense_data]
    expense_model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    expense_model.fit(expense_descriptions, expense_categories)
    logger.info("Huấn luyện mô hình CHI TIÊU thành công!")

    # --- Dữ liệu huấn luyện cho THU NHẬP (Mở rộng) ---
    income_data = [
        # Lương
        ("lương tháng 8", "Lương"), ("nhận lương công ty", "Lương"), ("lương tháng 7", "Lương"),
        ("lương part-time", "Lương"), ("nhận lương", "Lương"), ("ting ting lương về", "Lương"),
        # Thưởng
        ("thưởng dự án", "Thưởng"), ("được sếp thưởng", "Thưởng"), ("thưởng lễ", "Thưởng"),
        ("thưởng cuối năm", "Thưởng"), ("bonus", "Thưởng"), ("nhận tiền thưởng", "Thưởng"),
        # Thu nhập phụ
        ("tiền cho thuê xe", "Thu nhập phụ"), ("cho thuê nhà", "Thu nhập phụ"), ("bán đồ cũ online", "Thu nhập phụ"),
        ("tiền dạy thêm", "Thu nhập phụ"), ("làm freelancer", "Thu nhập phụ"), ("tiền cho thuê phòng", "Thu nhập phụ"),
        ("bán hàng online", "Thu nhập phụ"),
        # Đầu tư
        ("lãi ngân hàng", "Đầu tư"), ("lợi nhuận chứng khoán", "Đầu tư"), ("tiền cổ tức", "Đầu tư"),
        ("lãi tiết kiệm", "Đầu tư"),
        # Nguồn khác
        ("được cho tiền", "Khác"), ("quà mừng cưới", "Khác"), ("nhận tiền hoàn thuế", "Khác"),
        ("bố mẹ cho", "Khác"),
    ]
    income_descriptions = [item[0] for item in income_data]
    income_categories = [item[1] for item in income_data]
    income_model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    income_model.fit(income_descriptions, income_categories)
    logger.info("Huấn luyện mô hình THU NHẬP thành công!")

def classify_transaction(description: str, transaction_type: str) -> str:
    """Phân loại giao dịch dựa trên loại (thu/chi) bằng mô hình ML."""
    if transaction_type == 'chi' and expense_model:
        return expense_model.predict([description])[0]
    elif transaction_type == 'thu' and income_model:
        return income_model.predict([description])[0]
    return "Khác"

# --- CÁC HÀM XỬ LÝ LỆNH ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gửi tin nhắn chào mừng khi người dùng gõ /start."""
    user = update.effective_user
    await update.message.reply_html(
        f"Xin chào {user.mention_html()}!\n\n"
        f"Tôi là бот quản lý Thu - Chi cá nhân.\n\n"
        f"✍️ **Để ghi một giao dịch, chỉ cần gõ số tiền và nội dung.**\n"
        f"<b>Ví dụ chi tiêu:</b> <code>50000 ăn trưa</code>\n"
        f"<b>Ví dụ thu nhập:</b> <code>thu 10000000 lương</code>\n\n"
        f"Gõ /help để xem tất cả các lệnh.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hiển thị thông tin trợ giúp và các lệnh có sẵn."""
    await update.message.reply_text(
        "💡 **Các lệnh bạn có thể dùng:**\n\n"
        "/start - Bắt đầu và xem hướng dẫn\n"
        "/help - Xem lại tin nhắn này\n"
        "/tuan - Thống kê Thu-Chi tuần này\n"
        "/thang - Thống kê Thu-Chi tháng này\n"
        "/thongke - Vẽ biểu đồ chi tiêu tháng này"
    )

async def handle_transaction_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý tin nhắn giao dịch một cách linh hoạt."""
    text = update.message.text.lower()
    user_id = update.message.from_user.id
    numbers = re.findall(r'\d+', text)
    if not numbers:
        await update.message.reply_text('⚠️ Không tìm thấy số tiền trong tin nhắn của bạn.')
        return
    amount = max([int(n) for n in numbers])
    description_full = text
    for num_str in numbers:
        description_full = re.sub(r'\b' + num_str + r'\b', '', description_full).strip()
    transaction_type = 'chi'
    icon = '💸'
    if any(keyword in description_full.split() for keyword in ['thu', '+', 'nhận', 'lương', 'thưởng', 'bonus', 'lãi']):
        transaction_type = 'thu'
        icon = '💰'
    keywords_to_remove = ['thu', 'chi', '+', '-']
    description_words = [word for word in description_full.split() if word not in keywords_to_remove]
    description = ' '.join(description_words).strip()
    if not description:
        await update.message.reply_text('⚠️ Giao dịch của bạn cần có nội dung mô tả.', parse_mode='HTML')
        return
    category = classify_transaction(description, transaction_type)
    date = datetime.now()
    save_transaction(user_id, date, amount, category, description, transaction_type)
    await update.message.reply_text(
        f'{icon} Đã ghi nhận một khoản <b>{transaction_type.upper()}</b>\n'
        f'<b>Số tiền:</b> {amount:,.0f} VNĐ\n'
        f'<b>Nội dung:</b> {description}\n'
        f'<b>Danh mục:</b> {category}',
        parse_mode='HTML'
    )

async def weekly_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Thống kê thu chi trong tuần hiện tại."""
    now = datetime.now()
    await generate_full_report(update, context, "week", f"Tuần {now.isocalendar().week}, {now.year}")

async def monthly_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Thống kê thu chi trong tháng hiện tại."""
    now = datetime.now()
    await generate_full_report(update, context, "month", f"Tháng {now.month}/{now.year}")

async def thongke_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tạo và gửi biểu đồ thống kê chi tiêu tháng hiện tại."""
    user_id = update.message.from_user.id
    now = datetime.now()
    
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        await update.message.reply_text('Bạn chưa có dữ liệu nào để thống kê.')
        return

    # Lọc dữ liệu chi tiêu của người dùng trong tháng hiện tại
    expense_df = df[
        (df['user_id'] == user_id) &
        (df['type'] == 'chi') &
        (df['date'].dt.month == now.month) &
        (df['date'].dt.year == now.year)
    ]

    if expense_df.empty:
        await update.message.reply_text('Bạn không có chi tiêu nào trong tháng này để vẽ biểu đồ.')
        return

    # Nhóm theo danh mục và tính tổng
    category_stats = expense_df.groupby('category')['amount'].sum()

    # Vẽ biểu đồ
    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))

    labels = category_stats.index
    sizes = category_stats.values

    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, textprops=dict(color="w"))
    
    ax.legend(wedges, labels,
              title="Danh mục",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
              
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(f"Biểu đồ Chi tiêu Tháng {now.month}/{now.year}")

    # Lưu biểu đồ vào một buffer trong bộ nhớ
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # Gửi ảnh cho người dùng
    await update.message.reply_photo(photo=buf, caption="Đây là biểu đồ phân tích chi tiêu tháng này của bạn.")


# --- CÁC HÀM TIỆN ÍCH ---

def save_transaction(user_id, date, amount, category, description, transaction_type):
    """Lưu giao dịch (thu hoặc chi) vào file CSV."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['user_id', 'date', 'type', 'amount', 'category', 'description'])
    new_row = pd.DataFrame([{'user_id': user_id, 'date': date.strftime('%Y-%m-%d %H:%M:%S'), 'type': transaction_type, 'amount': amount, 'category': category, 'description': description}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

async def generate_full_report(update: Update, context: ContextTypes.DEFAULT_TYPE, period_type: str, period_name: str):
    """Tạo và gửi báo cáo Thu-Chi tổng hợp cho một khoảng thời gian."""
    user_id = update.message.from_user.id
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        await update.message.reply_text('Bạn chưa có dữ liệu nào để thống kê.')
        return
    now = datetime.now()
    if period_type == "week":
        period_filter = (df['date'].dt.isocalendar().week == now.isocalendar().week) & (df['date'].dt.year == now.year)
    else:
        period_filter = (df['date'].dt.month == now.month) & (df['date'].dt.year == now.year)
    user_df = df[(df['user_id'] == user_id) & period_filter]
    if user_df.empty:
        await update.message.reply_text(f'Bạn không có giao dịch nào trong {period_name.lower()}.')
        return
    income_df = user_df[user_df['type'] == 'thu']
    expense_df = user_df[user_df['type'] == 'chi']
    total_income = income_df['amount'].sum()
    total_expense = expense_df['amount'].sum()
    balance = total_income - total_expense
    response = f"📊 <b>Báo cáo tài chính {period_name}</b>\n\n"
    response += f"🟢 <b>Tổng Thu:</b> {total_income:,.0f} VNĐ\n"
    response += f"🔴 <b>Tổng Chi:</b> {total_expense:,.0f} VNĐ\n"
    response += f"<b>⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯</b>\n"
    response += f"📈 <b>Số dư: {balance:,.0f} VNĐ</b>\n\n"
    if not income_df.empty:
        response += "<b>--- Chi tiết các khoản THU ---</b>\n"
        income_stats = income_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        for category, amount in income_stats.items():
            response += f"  - {category}: {amount:,.0f} VNĐ\n"
        response += "\n"
    if not expense_df.empty:
        response += "<b>--- Chi tiết các khoản CHI ---</b>\n"
        expense_stats = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        for category, amount in expense_stats.items():
            response += f"  - {category}: {amount:,.0f} VNĐ\n"
    await update.message.reply_text(response, parse_mode='HTML')

# --- HÀM MAIN ĐỂ KHỞI ĐỘNG BOT ---

def main() -> None:
    """Khởi động bot và lắng nghe các yêu cầu."""
    if not TOKEN:
        logger.error("LỖI: Không tìm thấy TELEGRAM_API_TOKEN trong file .env")
        return
    train_models()
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tuan", weekly_stats_command))
    application.add_handler(CommandHandler("thang", monthly_stats_command))
    application.add_handler(CommandHandler("thongke", thongke_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_transaction_message))
    print("Bot Quản lý Thu-Chi (phiên bản vẽ biểu đồ) đang chạy...")
    application.run_polling()
    print("Bot đã dừng.")

if __name__ == '__main__':
    main()
