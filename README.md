# 🤖 Bot Quản Lý Chi Tiêu Cá Nhân (Telegram)

Chào mừng bạn đến với **Bot Quản Lý Thu Chi Cá Nhân**!  
Dự án này giúp bạn ghi chép, phân tích, thống kê tài chính cá nhân ngay trên Telegram, ứng dụng AI để tự động phân loại giao dịch và học hỏi từ chính bạn.

---

## 📚 Mục lục
1. [🌟 Tính năng nổi bật](#-tính-năng-nổi-bật)
2. [🖥️ Yêu cầu hệ thống](#️-yêu-cầu-hệ-thống)
3. [📦 Cấu trúc dự án](#-cấu-trúc-dự-án)
4. [⚡ Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
5. [📝 Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
6. [🤔 FAQ - Câu hỏi thường gặp](#-faq---câu-hỏi-thường-gặp)
7. [📬 Liên hệ & Đóng góp](#-liên-hệ--đóng-góp)

---

## 🌟 Tính năng nổi bật

- 💬 **Giao tiếp tự nhiên:** Ghi giao dịch bằng tin nhắn như "50000 ăn trưa", bot tự nhận diện số tiền, loại giao dịch, phân loại danh mục.
- 🤖 **AI/Machine Learning:** Bot sử dụng mô hình học máy để phân loại giao dịch, tự động học thêm từ dữ liệu bạn cung cấp.
- 📊 **Thống kê & Báo cáo:** Xem báo cáo tuần/tháng, biểu đồ chi tiêu, thống kê chi tiết từng danh mục.
- 🏷️ **Tự dạy bot:** Bạn có thể dạy bot nhận diện các danh mục mới bằng lệnh /dayhoc.
- 🔒 **Bảo mật:** Dữ liệu lưu trữ cục bộ, không gửi lên server bên ngoài.
- 🛠️ **Dễ mở rộng:** Có thể huấn luyện lại mô hình, thêm dữ liệu, backup dễ dàng.

---

## 🖥️ Yêu cầu hệ thống

- Python >= 3.8
- Các thư viện: `python-telegram-bot`, `pandas`, `scikit-learn`, `matplotlib`, `python-dotenv`, `sqlite3`, v.v.
- Tài khoản Telegram & tạo bot với BotFather để lấy API Token.

---

## 📦 Cấu trúc dự án

```
.
├── main.py                # Mã nguồn chính của bot Telegram
├── finance_bot.db         # Cơ sở dữ liệu SQLite lưu giao dịch & dữ liệu học máy
├── expense_model.pkl      # Mô hình phân loại chi tiêu (tự động sinh)
├── income_model.pkl       # Mô hình phân loại thu nhập (tự động sinh)
├── .env                   # Thông tin cấu hình (API Token, Admin ID)
├── .gitignore
└── README.md
```

---

## ⚡ Hướng dẫn cài đặt

1. **Clone dự án:**
   ```bash
   git clone https://github.com/tenban/QuanLiChiTieu.git
   cd QuanLiChiTieu
   ```

2. **Cài đặt thư viện:**

   ```bash
   pip install -r requirements.txt
   ```

   *(Nếu chưa có file requirements.txt, hãy cài các thư viện sau: python-telegram-bot, pandas, scikit-learn, matplotlib, python-dotenv)*

3. **Tạo file `.env`:**

   ```
   TELEGRAM_API_TOKEN=your_telegram_bot_token
   ADMIN_USER_ID=your_telegram_user_id
   ```

4. **Chạy bot:**

   ```bash
   python main.py
   ```

   Bot sẽ tự động tạo database và các file mô hình nếu chưa có.

---

## 📝 Hướng dẫn sử dụng

### 1. **Ghi giao dịch**

- Gửi tin nhắn cho bot:  
  `50000 ăn trưa`  
  `120000 đổ xăng`
- Bot sẽ tự động nhận diện số tiền, mô tả, loại giao dịch (thu/chi), phân loại danh mục.

### 2. **Xem thống kê**

- `/tuan` — Thống kê thu chi tuần này
- `/thang` — Thống kê thu chi tháng này
- `/thongke` — Biểu đồ chi tiêu tháng này

### 3. **Dạy bot phân loại mới**

- `/dayhoc chi Đi lại : phí gửi xe tháng`
- `/dayhoc thu Lương : lương tháng 5`
- Sau đó, admin có thể dùng `/hoclai` để bot học lại mô hình mới.

### 4. **Xem trợ giúp**

- `/help` — Xem tất cả lệnh hỗ trợ

---

## 🤔 FAQ - Câu hỏi thường gặp

**1. Bot có lưu dữ liệu của tôi lên server không?**  
> ❌ Không! Dữ liệu chỉ lưu trên máy chủ chạy bot.

**2. Có thể dùng bot cho nhiều người không?**  
> ✔️ Có! Mỗi người dùng Telegram sẽ có dữ liệu riêng biệt.

**3. Làm sao để bot phân loại đúng danh mục?**  
> ✔️ Bot sẽ học dần từ dữ liệu bạn nhập và các lệnh /dayhoc.

**4. Tôi muốn backup dữ liệu?**  
> ✔️ Chỉ cần sao lưu file `finance_bot.db` và các file mô hình `.pkl`.

**5. Không thấy file requirements.txt?**  
> Bạn có thể tự tạo hoặc cài các thư viện cần thiết như hướng dẫn ở trên.

---

## 📬 Liên hệ & Đóng góp

- Nếu bạn có ý tưởng, góp ý hoặc phát hiện lỗi, hãy tạo issue hoặc pull request trên GitHub.
- Email: [giangpt@duck.com](mailto:giangpt@duck.com)

---

> Chúc bạn quản lý tài chính hiệu quả, chủ động cho tương lai! 💰🌱 