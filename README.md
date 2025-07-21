# 💸 Quản Lý Chi Tiêu Cá Nhân

Xin chào! 👋 Đây là dự án **Bảng điều khiển tài chính cá nhân** giúp bạn chủ động kiểm soát thu chi, phân tích tài chính và hướng tới cuộc sống cân bằng hơn.

---

## 📚 Mục lục
1. [🌟 Tính năng nổi bật](#-tính-năng-nổi-bật)
2. [🖥️ Yêu cầu hệ thống](#️-yêu-cầu-hệ-thống)
3. [⚡ Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
4. [📝 Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
5. [❓ FAQ - Câu hỏi thường gặp](#-faq---câu-hỏi-thường-gặp)
6. [📬 Liên hệ & Đóng góp](#-liên-hệ--đóng-góp)

---

## 🌟 Tính năng nổi bật
- 🚀 **Trực quan, dễ dùng:** Giao diện hiện đại, thao tác kéo-thả file CSV cực nhanh.
- 📈 **Thống kê tức thì:** Xem tổng thu, tổng chi, số dư, biểu đồ phân loại chi tiêu, so sánh thu/chi.
- 🗓️ **Lọc thời gian linh hoạt:** Xem dữ liệu theo tuần, tháng hoặc toàn bộ.
- 🧾 **Lịch sử giao dịch:** Bảng chi tiết, dễ tra cứu, sắp xếp.
- 🔒 **Bảo mật:** Dữ liệu chỉ xử lý trên trình duyệt, không gửi lên server.

---

## 🖥️ Yêu cầu hệ thống
- Trình duyệt hiện đại (Chrome, Edge, Firefox, Safari...)
- Không cần cài đặt backend, không cần internet sau khi tải về.

---

## ⚡ Hướng dẫn cài đặt

1. **Tải mã nguồn:**
   ```bash
   git clone https://github.com/tenban/QuanLiChiTieu.git
   ```
   Hoặc tải file ZIP về và giải nén.

2. **Chuẩn bị file dữ liệu:**
   - Tạo file `thu_chi_data.csv` với cấu trúc:
     | date       | description      | category   | amount   | type |
     |------------|------------------|------------|----------|------|
     | 2024-05-01 | Lương tháng 5    | Lương      | 15000000 | thu  |
     | 2024-05-02 | Mua sắm          | Mua sắm    | 500000   | chi  |
   - Các trường:
     - `date`: Ngày giao dịch (YYYY-MM-DD)
     - `description`: Nội dung
     - `category`: Danh mục
     - `amount`: Số tiền
     - `type`: "thu" hoặc "chi"

---

## 📝 Hướng dẫn sử dụng

1. **Mở file `index.html`** bằng trình duyệt (double click hoặc chuột phải chọn "Open with Browser").
2. **Tải lên file CSV:**
   - Nhấn vào vùng tải lên hoặc kéo-thả file `thu_chi_data.csv` vào.
3. **Khám phá dữ liệu:**
   - Xem các số liệu tổng quan, biểu đồ, lịch sử giao dịch.
   - Dùng bộ lọc thời gian để phân tích sâu hơn.
4. **Lưu ý:**
   - Nếu có lỗi, kiểm tra lại định dạng file CSV.
   - Dữ liệu KHÔNG bị lưu trữ lên mạng.

---

## ❓ FAQ - Câu hỏi thường gặp

**1. Tôi có thể dùng trên điện thoại không?**  
> ✔️ Có! Ứng dụng tương thích tốt với trình duyệt di động.

**2. File CSV phải đúng định dạng nào?**  
> ✔️ Đúng thứ tự cột: `date, description, category, amount, type`.

**3. Có thể thêm nhiều loại danh mục không?**  
> ✔️ Bạn có thể tự do đặt tên danh mục trong cột `category`.

**4. Dữ liệu của tôi có bị gửi lên mạng không?**  
> ❌ Không! Mọi xử lý đều thực hiện trên máy bạn.

---

## 📬 Liên hệ & Đóng góp
- Nếu bạn có ý tưởng, góp ý hoặc phát hiện lỗi, hãy tạo issue hoặc pull request trên GitHub.
- Email: [giangpt@duck.com](mailto:giangpt@duck.com)

---

> Chúc bạn quản lý tài chính hiệu quả, chủ động cho tương lai! 💰🌱 