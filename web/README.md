# Web Điều Khiển & Giám Sát Xe Tự Hành

Giao diện Web điều khiển xe qua **chung mạng Wi-Fi** hoặc **cáp LAN**, tự động thích ứng cho cả **Máy tính** và **Điện thoại**.

## Cấu trúc gọn nhẹ
- `index.html`, `css/style.css`, `js/app.js`: Giao diện điều khiển & giám sát (tự động PC/Mobile).
- `server.py`: Máy chủ Web tích hợp sẵn vào `RobotMain.py` (hoặc chạy riêng nếu muốn test giao diện khi chưa bật xe).

## Cách chạy trên Raspberry Pi (chỉ chạy 1 file `RobotMain.py`)
```bash
python3 RobotMain.py
```
`RobotMain.py` tự bật sẵn Web Server trên cổng `8080` và in ra địa chỉ LAN/Wi-Fi. Từ **bất kỳ máy tính hay điện thoại nào** cùng mạng, mở:
- `http://xetuhanh.local:8080` (nếu đặt hostname Pi là `xetuhanh`) hoặc `http://<IP-cua-Pi>:8080`.

Để Pi tự chạy `RobotMain.py` mỗi khi bật nguồn (không cần chạy tay lại), chỉ cần thêm `RobotMain.py --headless` vào `systemd` hoặc `crontab -e` (`@reboot sleep 10 && cd /home/pi/XeTuHanh && python3 RobotMain.py --headless`).
