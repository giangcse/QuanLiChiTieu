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

# --- CẤU HÌNH ---
TOKEN = 'YOUR_TELEGRAM_API_TOKEN' # THAY THẾ BẰNG TOKEN CỦA BẠN
DATA_FILE = 'thu_chi_data.csv' # Đổi tên file để lưu cả thu và chi

# Bật logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- BỘ PHẬN MÁY HỌC ---
# Hai mô hình riêng biệt cho Chi tiêu và Thu nhập
expense_model = None
income_model = None

def train_models():
    """Huấn luyện đồng thời cả hai mô hình phân loại."""
    global expense_model, income_model
    logger.info("Bắt đầu huấn luyện các mô hình...")

    # --- Dữ liệu huấn luyện cho CHI TIÊU ---
    expense_data = [
        ("cơm sườn trưa", "Ăn uống"), ("mua ly trà sữa", "Ăn uống"), ("cà phê với bạn", "Ăn uống"),
        ("đổ xăng xe máy", "Đi lại"), ("vé xe bus tháng", "Đi lại"), ("tiền grab đi làm", "Đi lại"),
        ("mua áo sơ mi", "Mua sắm"), ("đặt hàng shopee", "Mua sắm"), ("mua một đôi giày mới", "Mua sắm"),
        ("thanh toán tiền điện", "Hóa đơn"), ("đóng tiền net FPT", "Hóa đơn"), ("tiền nhà tháng 8", "Hóa đơn"),
        ("vé xem phim cgv", "Giải trí"), ("mua thuốc cảm", "Sức khỏe"),
    ]
    expense_descriptions = [item[0] for item in expense_data]
    expense_categories = [item[1] for item in expense_data]
    
    expense_model = Pipeline([
        ('tfidf', TfidfVectorizer()), ('clf', MultinomialNB()),
    ])
    expense_model.fit(expense_descriptions, expense_categories)
    logger.info("Huấn luyện mô hình CHI TIÊU thành công!")

    # --- Dữ liệu huấn luyện cho THU NHẬP ---
    income_data = [
        ("lương tháng 8", "Lương"), ("nhận lương công ty", "Lương"),
        ("thưởng dự án", "Thưởng"), ("được sếp thưởng", "Thưởng"),
        ("tiền cho thuê xe", "Thu nhập phụ"), ("cho thuê nhà", "Thu nhập phụ"),
        ("bán đồ cũ online", "Thu nhập phụ"), ("lãi ngân hàng", "Đầu tư"),
    ]
    income_descriptions = [item[0] for item in income_data]
    income_categories = [item[1] for item in income_data]

    income_model = Pipeline([
        ('tfidf', TfidfVectorizer()), ('clf', MultinomialNB()),
    ])
    income_model.fit(income_descriptions, income_categories)
    logger.info("Huấn luyện mô hình THU NHẬP thành công!")

def classify_transaction(description: str, transaction_type: str) -> str:
    """Phân loại giao dịch dựa trên loại (thu/chi)."""
    if transaction_type == 'chi' and expense_model:
        return expense_model.predict([description])[0]
    elif transaction_type == 'thu' and income_model:
        return income_model.predict([description])[0]
    return "Khác"

# --- CÁC HÀM XỬ LÝ LỆNH ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        f"Xin chào {user.mention_html()}!\n\n"
        f"Tôi là бот quản lý Thu - Chi cá nhân.\n\n"
        f"✍️ **Để ghi một giao dịch, chỉ cần gõ số tiền và nội dung.**\n"
        f"Bạn có thể đặt số tiền ở trước hoặc sau nội dung.\n\n"
        f"<b>Ví dụ chi tiêu:</b>\n"
        f"<code>50000 ăn trưa</code> hoặc <code>ăn trưa 50000</code>\n\n"
        f"<b>Ví dụ thu nhập (thêm 'thu' hoặc '+'):</b>\n"
        f"<code>thu 10000000 lương</code> hoặc <code>lương 10000000 +</code>\n\n"
        f"<i>Nếu không có 'thu' hoặc '+', tôi sẽ mặc định là CHI TIÊU.</i>\n\n"
        f"Gõ /help để xem tất cả các lệnh.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "💡 **Các lệnh bạn có thể dùng:**\n\n"
        "/start - Bắt đầu và xem hướng dẫn\n"
        "/help - Xem lại tin nhắn này\n"
        "/tuan - Thống kê Thu-Chi tuần này\n"
        "/thang - Thống kê Thu-Chi tháng này"
    )

async def handle_transaction_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Xử lý tin nhắn giao dịch một cách linh hoạt.
    Tự động phát hiện số tiền và nội dung.
    """
    text = update.message.text.lower()
    user_id = update.message.from_user.id
    
    # Bước 1: Tìm tất cả các số trong tin nhắn
    numbers = re.findall(r'\d+', text)

    # Bước 2: Kiểm tra nếu không có số tiền
    if not numbers:
        await update.message.reply_text(
            '⚠️ Không tìm thấy số tiền trong tin nhắn của bạn. Vui lòng thử lại.'
        )
        return
    
    # Bước 3: Xác định số tiền (giả định là số lớn nhất)
    amount = max([int(n) for n in numbers])
    
    # Bước 4: Tách nội dung và các từ khóa
    # Xóa tất cả các số đã tìm thấy khỏi văn bản gốc để có được nội dung
    description_full = text
    for num_str in numbers:
        # Sử dụng regex với \b (word boundary) để đảm bảo chỉ xóa các từ là số
        description_full = re.sub(r'\b' + num_str + r'\b', '', description_full).strip()
    
    # Bước 5: Xác định loại giao dịch (thu/chi)
    transaction_type = 'chi'  # Mặc định là chi tiêu
    icon = '💸'
    
    # Kiểm tra các từ khóa của thu nhập trong nội dung
    if any(keyword in description_full.split() for keyword in ['thu', '+']):
        transaction_type = 'thu'
        icon = '💰'
    
    # Bước 6: Làm sạch nội dung cuối cùng
    # Xóa các từ khóa (thu, chi, +, -) khỏi nội dung
    keywords_to_remove = ['thu', 'chi', '+', '-']
    description_words = [word for word in description_full.split() if word not in keywords_to_remove]
    description = ' '.join(description_words).strip()

    if not description:
        await update.message.reply_text(
            '⚠️ Giao dịch của bạn cần có nội dung mô tả.\n'
            'Ví dụ: <code>Ăn sáng 20000</code>',
            parse_mode='HTML'
        )
        return

    # Bước 7: Phân loại và lưu trữ (logic không đổi)
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

# --- CÁC HÀM TIỆN ÍCH ---

def save_transaction(user_id, date, amount, category, description, transaction_type):
    """Lưu giao dịch (thu hoặc chi) vào file CSV."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['user_id', 'date', 'type', 'amount', 'category', 'description'])

    new_row = pd.DataFrame([{
        'user_id': user_id,
        'date': date.strftime('%Y-%m-%d %H:%M:%S'),
        'type': transaction_type,
        'amount': amount,
        'category': category,
        'description': description
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

async def generate_full_report(update: Update, context: ContextTypes.DEFAULT_TYPE, period_type: str, period_name: str):
    """Tạo báo cáo Thu-Chi tổng hợp."""
    user_id = update.message.from_user.id
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        await update.message.reply_text('Bạn chưa có dữ liệu nào để thống kê.')
        return

    now = datetime.now()
    # Lọc dữ liệu theo thời gian (tuần hoặc tháng)
    if period_type == "week":
        period_filter = (df['date'].dt.isocalendar().week == now.isocalendar().week) & (df['date'].dt.year == now.year)
    else: # month
        period_filter = (df['date'].dt.month == now.month) & (df['date'].dt.year == now.year)

    user_df = df[(df['user_id'] == user_id) & period_filter]

    if user_df.empty:
        await update.message.reply_text(f'Bạn không có giao dịch nào trong {period_name.lower()}.')
        return

    # Tách thành 2 bảng thu và chi
    income_df = user_df[user_df['type'] == 'thu']
    expense_df = user_df[user_df['type'] == 'chi']

    total_income = income_df['amount'].sum()
    total_expense = expense_df['amount'].sum()
    balance = total_income - total_expense

    # Xây dựng chuỗi phản hồi
    response = f"📊 <b>Báo cáo tài chính {period_name}</b>\n\n"
    response += f"🟢 <b>Tổng Thu:</b> {total_income:,.0f} VNĐ\n"
    response += f"🔴 <b>Tổng Chi:</b> {total_expense:,.0f} VNĐ\n"
    response += f"<b>⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯</b>\n"
    response += f"📈 <b>Số dư: {balance:,.0f} VNĐ</b>\n\n"

    # Thống kê chi tiết thu nhập
    if not income_df.empty:
        response += "<b>--- Chi tiết các khoản THU ---</b>\n"
        income_stats = income_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        for category, amount in income_stats.items():
            response += f"  - {category}: {amount:,.0f} VNĐ\n"
        response += "\n"

    # Thống kê chi tiết chi tiêu
    if not expense_df.empty:
        response += "<b>--- Chi tiết các khoản CHI ---</b>\n"
        expense_stats = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
        for category, amount in expense_stats.items():
            response += f"  - {category}: {amount:,.0f} VNĐ\n"

    await update.message.reply_text(response, parse_mode='HTML')

# --- HÀM MAIN ĐỂ KHỞI ĐỘNG BOT ---

def main() -> None:
    """Khởi động bot và lắng nghe các yêu cầu."""
    # Huấn luyện cả 2 mô hình khi bot khởi động
    train_models()

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tuan", weekly_stats_command))
    application.add_handler(CommandHandler("thang", monthly_stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_transaction_message))

    print("Bot Quản lý Thu-Chi (phiên bản thông minh) đang chạy...")
    application.run_polling()
    print("Bot đã dừng.")

if __name__ == '__main__':
    main()
