# این یک پروژه برای ترجمه صدای یک ویدئو کلیپ با استفاده از moviepy می باشد
### برای راه اندازی پروژه ابتدا با uv پروژه را در یک فولدر راه اندازی می کنیم

```bash
uv init
uv venv
```
### حالا وارد محیط Virtual Environment پروژه می شویم
```bash
.\.venv\Scripts\activate
```

### حالا پکیج های مد نظرمان را با uv نصب میکنیم اگر در حین توسعه پکیج جدیدی خواستیم با دستور uv add اضافه میکنیم

```bash
uv add moviepy yt_dlp rich openai-whisper ollama
```
### حالا یک فایل برای شروع توسعه پروژه به اسم functions.py ایجاد می کنیم و کتابخانه های مورد نظرمان را import می کنیم

```bash
import moviepy
import yt_dlp
import whisper
import torch
import ollama
from rich import print

```

### برای پروژه های moviepy باید ffmpeg روی سیستم نصب باشد حالا با استفاده از دستورات ذیل صحت نصب آن را چک می می کنیم
```bash
print(moviepy.config.check())

```
###### که باید پاسخی شبیه به این را به ما برگرداند:
```bash
MoviePy: ffmpeg successfully found in '<project folder>\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'.
MoviePy: ffplay successfully found in 'ffplay'.

```