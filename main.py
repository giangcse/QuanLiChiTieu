# -*- coding: utf-8 -*-

import logging
import re
import pandas as pd
from datetime import datetime
from telegram import Update, ForceReply, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

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

# Thư viện cơ sở dữ liệu
import sqlite3

# Thư viện để lưu/tải mô hình
import pickle

# --- CẤU HÌNH ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_API_TOKEN')
DB_FILE = 'finance_bot.db'
EXPENSE_MODEL_FILE = 'expense_model.pkl'
INCOME_MODEL_FILE = 'income_model.pkl'
ADMIN_USER_ID = os.getenv('ADMIN_USER_ID')

# Bật logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- QUẢN LÝ CƠ SỞ DỮ LIỆU ---

def init_db():
    """Khởi tạo cơ sở dữ liệu và tạo các bảng nếu chưa tồn tại."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Bảng lưu giao dịch
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT,
            description TEXT
        )
    ''')
    # Bảng lưu các mẫu huấn luyện mới do người dùng cung cấp
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"Cơ sở dữ liệu '{DB_FILE}' đã được khởi tạo và kiểm tra.")

def save_transaction(user_id, date, amount, category, description, transaction_type):
    """Lưu giao dịch vào cơ sở dữ liệu SQLite."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transactions (user_id, date, type, amount, category, description) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, date.strftime('%Y-%m-%d %H:%M:%S'), transaction_type, amount, category, description)
    )
    conn.commit()
    conn.close()

def save_suggestion(user_id, type, category, description):
    """Lưu một gợi ý huấn luyện mới vào DB."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO training_suggestions (user_id, type, category, description) VALUES (?, ?, ?, ?)",
        (user_id, type, category, description)
    )
    conn.commit()
    conn.close()

def fetch_data_from_db(user_id, period_type='all'):
    """Lấy dữ liệu từ DB cho một người dùng và khoảng thời gian cụ thể."""
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT date, type, amount, category, description FROM transactions WHERE user_id = ?"
    now = datetime.now()
    params = [user_id]
    if period_type == 'month':
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        query += " AND date >= ?"
        params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
    elif period_type == 'week':
        start_of_week = now - pd.to_timedelta(now.weekday(), unit='d')
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        query += " AND date >= ?"
        params.append(start_of_week.strftime('%Y-%m-%d %H:%M:%S'))
    df = pd.read_sql_query(query, conn, params=tuple(params))
    conn.close()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


# --- BỘ PHẬN MÁY HỌC (Tối ưu hóa với Pickle và Học tăng cường) ---
expense_model = None
income_model = None

def train_models():
    """Tải hoặc huấn luyện lại mô hình, có kết hợp dữ liệu do người dùng cung cấp."""
    global expense_model, income_model

    def get_user_suggestions():
        """Lấy tất cả các gợi ý huấn luyện từ DB."""
        try:
            conn = sqlite3.connect(DB_FILE)
            df = pd.read_sql_query("SELECT type, category, description FROM training_suggestions", conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Không thể lấy dữ liệu huấn luyện từ DB: {e}")
            return pd.DataFrame()

    # --- Xử lý mô hình Chi tiêu ---
    if os.path.exists(EXPENSE_MODEL_FILE):
        with open(EXPENSE_MODEL_FILE, 'rb') as f:
            expense_model = pickle.load(f)
        logger.info(f"Đã tải mô hình CHI TIÊU từ file '{EXPENSE_MODEL_FILE}'.")
    else:
        logger.info(f"Không tìm thấy file mô hình '{EXPENSE_MODEL_FILE}'. Bắt đầu huấn luyện mới...")
        expense_data = [
            ("cơm sườn trưa", "Ăn uống"), ("mua ly trà sữa", "Ăn uống"), ("cà phê với bạn", "Ăn uống"),
            ("đổ xăng xe máy", "Đi lại"), ("vé xe bus tháng", "Đi lại"), ("tiền grab đi làm", "Đi lại"),
            ("mua áo sơ mi", "Mua sắm"), ("đặt hàng shopee", "Mua sắm"), ("mua một đôi giày mới", "Mua sắm"),
            ("thanh toán tiền điện", "Hóa đơn"), ("đóng tiền net FPT", "Hóa đơn"), ("tiền nhà tháng 8", "Hóa đơn"),
            ("vé xem phim cgv", "Giải trí"), ("mua thuốc cảm", "Sức khỏe"),
            ("học phí khóa học online", "Giáo dục"),
        ]
        
        user_suggestions_df = get_user_suggestions()
        user_expense_suggestions = user_suggestions_df[user_suggestions_df['type'] == 'chi']
        if not user_expense_suggestions.empty:
            additional_data = list(zip(user_expense_suggestions['description'], user_expense_suggestions['category']))
            expense_data.extend(additional_data)
            logger.info(f"Đã thêm {len(additional_data)} mẫu huấn luyện CHI TIÊU từ người dùng.")

        expense_descriptions = [item[0] for item in expense_data]
        expense_categories = [item[1] for item in expense_data]
        expense_model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
        expense_model.fit(expense_descriptions, expense_categories)
        logger.info("Huấn luyện mô hình CHI TIÊU thành công!")
        with open(EXPENSE_MODEL_FILE, 'wb') as f:
            pickle.dump(expense_model, f)
        logger.info(f"Đã lưu mô hình CHI TIÊU mới vào file '{EXPENSE_MODEL_FILE}'.")

    # --- Xử lý mô hình Thu nhập ---
    if os.path.exists(INCOME_MODEL_FILE):
        with open(INCOME_MODEL_FILE, 'rb') as f:
            income_model = pickle.load(f)
        logger.info(f"Đã tải mô hình THU NHẬP từ file '{INCOME_MODEL_FILE}'.")
    else:
        logger.info(f"Không tìm thấy file mô hình '{INCOME_MODEL_FILE}'. Bắt đầu huấn luyện mới...")
        income_data = [
            ("lương tháng 8", "Lương"), ("nhận lương công ty", "Lương"), ("thưởng dự án", "Thưởng"),
            ("tiền cho thuê xe", "Thu nhập phụ"), ("lãi ngân hàng", "Đầu tư"), ("bố mẹ cho", "Khác"),
        ]
        
        user_suggestions_df = get_user_suggestions() # Lấy lại nếu cần
        user_income_suggestions = user_suggestions_df[user_suggestions_df['type'] == 'thu']
        if not user_income_suggestions.empty:
            additional_data = list(zip(user_income_suggestions['description'], user_income_suggestions['category']))
            income_data.extend(additional_data)
            logger.info(f"Đã thêm {len(additional_data)} mẫu huấn luyện THU NHẬP từ người dùng.")

        income_descriptions = [item[0] for item in income_data]
        income_categories = [item[1] for item in income_data]
        income_model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
        income_model.fit(income_descriptions, income_categories)
        logger.info("Huấn luyện mô hình THU NHẬP thành công!")
        with open(INCOME_MODEL_FILE, 'wb') as f:
            pickle.dump(income_model, f)
        logger.info(f"Đã lưu mô hình THU NHẬP mới vào file '{INCOME_MODEL_FILE}'.")

def classify_transaction(description: str, transaction_type: str) -> str:
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
        f"✍️ **Ghi giao dịch:** <code>50000 ăn trưa</code>\n"
        f"✍️ **Dạy cho bot:** <code>/dayhoc chi Đi lại : phí gửi xe tháng</code>\n\n"
        f"Gõ /help để xem tất cả các lệnh.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "💡 **Các lệnh bạn có thể dùng:**\n\n"
        "/start - Bắt đầu\n"
        "/help - Xem lại tin nhắn này\n"
        "/tuan - Thống kê Thu-Chi tuần này\n"
        "/thang - Thống kê Thu-Chi tháng này\n"
        "/thongke - Vẽ biểu đồ chi tiêu tháng này\n"
        "/dayhoc - Dạy cho bot một nhãn mới.\n"
        "  Cú pháp: /dayhoc [thu/chi] [Tên danh mục] : [Nội dung mô tả]\n"
        "/hoclai - (Admin) Yêu cầu bot học lại từ đầu với dữ liệu mới."
    )

async def dayhoc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /dayhoc để thêm dữ liệu huấn luyện mới."""
    user_id = update.message.from_user.id
    text = update.message.text.split('/dayhoc', 1)[1].strip()

    # Regex để tách các thành phần: [thu/chi] [Danh mục] : [Mô tả]
    match = re.match(r'^(thu|chi)\s+([^:]+)\s*:\s*(.+)$', text, re.IGNORECASE)

    if not match:
        await update.message.reply_text(
            "⚠️ Cú pháp không đúng. Vui lòng sử dụng:\n"
            "<code>/dayhoc [thu/chi] [Tên danh mục] : [Nội dung mô tả]</code>\n\n"
            "<b>Ví dụ:</b> <code>/dayhoc chi Đi lại : phí đường bộ</code>",
            parse_mode='HTML'
        )
        return

    trans_type = match.group(1).lower()
    category = match.group(2).strip()
    description = match.group(3).strip()

    save_suggestion(user_id, trans_type, category, description)
    
    await update.message.reply_text(
        f"✅ Cảm ơn bạn! Tôi đã học được rằng:\n"
        f"  - <b>Loại:</b> {trans_type.upper()}\n"
        f"  - <b>Danh mục:</b> {category}\n"
        f"  - <b>Nội dung:</b> {description}\n\n"
        f"Dùng lệnh /hoclai (nếu bạn là admin) để cập nhật kiến thức cho bot.",
        parse_mode='HTML'
    )

async def hoclai_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """(Admin) Cho phép admin yêu cầu bot huấn luyện lại mô hình."""
    user_id = update.message.from_user.id
    
    if not ADMIN_USER_ID or user_id != int(ADMIN_USER_ID):
        await update.message.reply_text("⚠️ Bạn không có quyền thực hiện lệnh này.")
        return

    await update.message.reply_text("⏳ Bắt đầu quá trình huấn luyện lại mô hình... Vui lòng chờ một lát.")
    
    try:
        # Xóa các file mô hình cũ để buộc huấn luyện lại
        if os.path.exists(EXPENSE_MODEL_FILE):
            os.remove(EXPENSE_MODEL_FILE)
            logger.info(f"Đã xóa file mô hình cũ: {EXPENSE_MODEL_FILE}")
        if os.path.exists(INCOME_MODEL_FILE):
            os.remove(INCOME_MODEL_FILE)
            logger.info(f"Đã xóa file mô hình cũ: {INCOME_MODEL_FILE}")
        
        # Gọi lại hàm huấn luyện
        train_models()
        
        await update.message.reply_text("✅ Quá trình huấn luyện lại đã hoàn tất! Kiến thức mới đã được cập nhật.")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình huấn luyện lại: {e}")
        await update.message.reply_text(f"❌ Đã có lỗi xảy ra: {e}")

async def handle_transaction_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    words = description_full.split()
    transaction_type = None
    icon = ''
    if 'thu' in words or '+' in words:
        transaction_type = 'thu'
        icon = '💰'
    elif 'chi' in words or '-' in words:
        transaction_type = 'chi'
        icon = '💸'
    if transaction_type is None:
        if any(keyword in words for keyword in ['nhận', 'lương', 'thưởng', 'bonus', 'lãi']):
            transaction_type = 'thu'
            icon = '💰'
        else:
            transaction_type = 'chi'
            icon = '💸'
    keywords_to_remove = ['thu', 'chi', '+', '-']
    description_words = [word for word in words if word not in keywords_to_remove]
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
    now = datetime.now()
    await generate_full_report(update, context, "week", f"Tuần {now.isocalendar().week}, {now.year}")

async def monthly_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    now = datetime.now()
    await generate_full_report(update, context, "month", f"Tháng {now.month}/{now.year}")

async def thongke_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    now = datetime.now()
    user_df = fetch_data_from_db(user_id, 'month')
    expense_df = user_df[user_df['type'] == 'chi']
    if expense_df.empty:
        await update.message.reply_text('Bạn không có chi tiêu nào trong tháng này để vẽ biểu đồ.')
        return
    category_stats = expense_df.groupby('category')['amount'].sum()
    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
    labels = category_stats.index
    sizes = category_stats.values
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, textprops=dict(color="w"))
    ax.legend(wedges, labels, title="Danh mục", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(f"Biểu đồ Chi tiêu Tháng {now.month}/{now.year}")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    await update.message.reply_photo(photo=buf, caption="Đây là biểu đồ phân tích chi tiêu tháng này của bạn.")

async def generate_full_report(update: Update, context: ContextTypes.DEFAULT_TYPE, period_type: str, period_name: str):
    user_id = update.message.from_user.id
    user_df = fetch_data_from_db(user_id, period_type)
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
    
    init_db()
    train_models()
    
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tuan", weekly_stats_command))
    application.add_handler(CommandHandler("thang", monthly_stats_command))
    application.add_handler(CommandHandler("thongke", thongke_command))
    application.add_handler(CommandHandler("dayhoc", dayhoc_command))
    application.add_handler(CommandHandler("hoclai", hoclai_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_transaction_message))
    
    print("Bot Quản lý Thu-Chi (phiên bản Học tăng cường) đang chạy...")
    application.run_polling()
    print("Bot đã dừng.")

if __name__ == '__main__':
    main()
