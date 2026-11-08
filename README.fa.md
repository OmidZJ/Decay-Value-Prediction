# پیش‌بینی مقدار واپاشی

[English README](README.md)

مدل XGBoost لگاریتم مبنای ۱۰ نسبت شاخه‌ای واپاشی را پیش‌بینی می‌کند. در بهترین اجرای ثبت‌شده، مدل از یک دیاگرام منتخب، تابع زیان شبه‌هابر با `huber_slope = 0.8` و ۱۰۰ آزمون Optuna در هر بخش بیرونی استفاده کرد. مقدار $R^2$ برای پیش‌بینی‌های خارج از بخش آموزش روی تمام ۳٬۴۴۹ ردیف، **0.80143** بود.

## اجرا

فایل‌های `All PDG Training Decays 2026.csv` و `All PDG Decays 2026.csv` را کنار اسکریپت قرار دهید. وابستگی‌ها را نصب کنید و تنظیمات اجرای ۱۰۰ آزمون با GPU را از پوشهٔ اصلی مخزن اجرا کنید.

```bash
pip install numpy pandas xgboost==3.2.0 optuna==4.9.0 scikit-learn==1.6.1
python BDecay_XGBoost_NewPDG_All_Unfiltered_Optuna.py --device cuda --trials 100 --output xgb_new_pdg_gpu100
```

برای اجرای پیش‌فرض با CPU و ۴۰ آزمون، اسکریپت را بدون آرگومان اجرا کنید. در Kaggle همان دو فایل CSV را اضافه کنید، GPU را فعال کنید و [اسکریپت مستقل](BDecay_XGBoost_NewPDG_All_Unfiltered_Optuna.py) را در یک سلول کد قرار دهید. در `Config`، مقدار `n_trials` را روی `100` و `device` را روی `'cuda'` بگذارید. خروجی‌ها شامل پیش‌بینی‌های داده‌های کنارگذاشته‌شده، تقسیم‌بندی ردیف‌ها، معیارهای ارزیابی و مدل نهایی است.

## نتایج

| اجرا | تعداد آزمون در هر بخش بیرونی | پارامتر شبه‌هابر | $R^2$ خارج از بخش آموزش | RMSE (dex) | MAE (dex) |
| --- | ---: | --- | ---: | ---: | ---: |
| اجرای پایه با CPU | 40 | ثابت روی 0.8 | 0.80135 | 0.69582 | 0.47454 |
| اجرای کنترل با GPU | 40 | ثابت روی 0.8 | 0.79897 | 0.69999 | 0.47596 |
| **جست‌وجوی طولانی‌تر با GPU** | **100** | **ثابت روی 0.8** | **0.80143** | **0.69569** | **0.47157** |
| تنظیم پارامتر شبه‌هابر با GPU | 100 | تنظیم‌شده در بخش‌های داخلی | 0.79799 | 0.70168 | 0.47417 |

اجرای ۱۰۰ آزمون با پارامتر ثابت، $R^2$ را نسبت به اجرای کنترل روی همان GPU به اندازهٔ ۰٫۰۰۲۴۶ افزایش داد. اجرای کنترل از ۴۰ آزمون اول همان جست‌وجو استفاده کرد. افزایش نسبت به اجرای پایه با CPU حدود ۰٫۰۰۰۰۷ بود.

### مقایسهٔ اجراهای CPU روی همهٔ ردیف‌ها و روی واپاشی‌های B

هر دو اجرا با CPU، ۴۰ آزمون Optuna در هر بخش بیرونی، همان ۱٬۲۸۵ ویژگی مبتنی بر دیاگرام منتخب و `huber_slope = 0.8` انجام شدند. در اجرای مخصوص B، آموزش و اعتبارسنجی داخلی به مزون‌ها و باریون‌های دارای محتوای خالص باتمِ غیرصفر محدود شد. هر دو مدل روی همان ۷۵۱ ردیف B با تقسیم‌بندی بیرونی یکسان ارزیابی شدند. هر ۱۲۰ ردیف علامت‌گذاری‌شده در این گروه هم حفظ شدند.

| داده‌های آموزش | ردیف‌های کنارگذاشته‌شده برای ارزیابی | $R^2$ خارج از بخش آموزش | RMSE (dex) | MAE (dex) |
| --- | --- | ---: | ---: | ---: |
| فقط واپاشی‌های B (751 ردیف) | 751 ردیف B | 0.74609 | 0.63591 | 0.46307 |
| همهٔ ذرات (3,449 ردیف) | همهٔ 3,449 ردیف | 0.80135 | 0.69582 | 0.47454 |
| همهٔ ذرات (3,449 ردیف) | همان 751 ردیف B | **0.75530** | **0.62427** | **0.45879** |

دو ردیف آخر از پیش‌بینی‌های یک اجرای واحد با داده‌های همهٔ ذرات به دست آمده‌اند. روی همان ۷۵۱ ردیف B، آموزش با داده‌های همهٔ ذرات مقدار $R^2$ را به اندازهٔ ۰٫۰۰۹۲۱ افزایش داد. اجرای مخصوص B، با احتساب جست‌وجوی جداگانه برای ساخت مدل نهایی، ۱۲٫۲ دقیقه طول کشید.

#### ردیف‌های B چگونه برای ارزیابی کنار گذاشته می‌شوند؟

در هر بخش بیرونی، مدل همهٔ ذرات از ابتدا و در یک مرحله با ردیف‌های B و ردیف‌های سایر واپاشی‌ها در بخش آموزش همان تقسیم‌بندی آموزش می‌بیند. مدل مخصوص B از همان ردیف‌های B در بخش آموزش استفاده می‌کند. آموزش مدل و جست‌وجوی Optuna در هر دو حالت فقط از داده‌های بخش آموزش همان تقسیم‌بندی استفاده می‌کنند.

| نمونه برای بخش بیرونی اول | ردیف‌های آموزش | ردیف‌های کنارگذاشته‌شده برای محاسبهٔ امتیاز B |
| --- | --- | --- |
| مدل مخصوص B | ۶۰۶ ردیف B | ۱۴۵ ردیف B |
| مدل همهٔ ذرات | همان ۶۰۶ ردیف B + ۲٬۱۵۳ ردیف از سایر واپاشی‌ها | همان ۱۴۵ ردیف B |

برای هر مدل، هر ردیف B یک پیش‌بینی خارج از بخش آموزش دارد. امتیازهای B از مجموع این پیش‌بینی‌ها در هر پنج بخش بیرونی محاسبه می‌شوند.

## ستون‌های استفاده‌شده از فایل‌های CSV

### `All PDG Training Decays 2026.csv`

**ویژگی‌های واپاشی**

```text
Decay Type
Is Decay Radiative
Decay Q Value
Decay P*
```

**ورودی‌های دیاگرام**

```text
Diagram 1 Topology
Diagram 1 Quark Transition
Diagram 1 Effective CKM Matrix Element
Diagram 1 Relative Amplitude
Diagram 2 Topology
Diagram 2 Quark Transition
Diagram 2 Effective CKM Matrix Element
Diagram 2 Relative Amplitude
Diagram 3 Topology
Diagram 3 Quark Transition
Diagram 3 Effective CKM Matrix Element
Diagram 3 Relative Amplitude
```

**ذرهٔ اولیه**

```text
Initial Particle Multiplicity
Initial Particle Type
Initial Particle Mass
Initial Particle Width
Initial Particle Charge
Initial Particle Quantum C
Initial Particle Quantum G
Initial Particle Quantum I
Initial Particle Quantum J
Initial Particle Quantum P
Initial Particle Structure As Int
```

**ذرهٔ میانی** (نام ستون‌ها مطابق املای فایل CSV است)

```text
Intermidiate Particle Multiplicity
Intermidiate Particle Type
Intermidiate Particle Mass
Intermidiate Particle Width
Intermidiate Particle Charge
Intermidiate Particle Quantum C
Intermidiate Particle Quantum G
Intermidiate Particle Quantum I
Intermidiate Particle Quantum J
Intermidiate Particle Quantum P
Intermidiate Particle Structure As Int
```

**ذرهٔ محصول ۱**

```text
Product Particle 1 Multiplicity
Product Particle 1 Type
Product Particle 1 Mass
Product Particle 1 Width
Product Particle 1 Charge
Product Particle 1 Quantum C
Product Particle 1 Quantum G
Product Particle 1 Quantum I
Product Particle 1 Quantum J
Product Particle 1 Quantum P
Product Particle 1 Structure As Int
```

**ذرهٔ محصول ۲**

```text
Product Particle 2 Multiplicity
Product Particle 2 Type
Product Particle 2 Mass
Product Particle 2 Width
Product Particle 2 Charge
Product Particle 2 Quantum C
Product Particle 2 Quantum G
Product Particle 2 Quantum I
Product Particle 2 Quantum J
Product Particle 2 Quantum P
Product Particle 2 Structure As Int
```

**ذرهٔ محصول ۳**

```text
Product Particle 3 Multiplicity
Product Particle 3 Type
Product Particle 3 Mass
Product Particle 3 Width
Product Particle 3 Charge
Product Particle 3 Quantum C
Product Particle 3 Quantum G
Product Particle 3 Quantum I
Product Particle 3 Quantum J
Product Particle 3 Quantum P
Product Particle 3 Structure As Int
```

**ذرهٔ محصول ۴**

```text
Product Particle 4 Multiplicity
Product Particle 4 Type
Product Particle 4 Mass
Product Particle 4 Width
Product Particle 4 Charge
Product Particle 4 Quantum C
Product Particle 4 Quantum G
Product Particle 4 Quantum I
Product Particle 4 Quantum J
Product Particle 4 Quantum P
Product Particle 4 Structure As Int
```

تعداد ذرات محصول تعیین می‌کند هر ذرهٔ تکراری چند بار در ورودی قرار بگیرد. اعتبار تعداد ذرات اولیه و میانی نیز بررسی می‌شود. از ستون‌های `Structure As Int` تعداد خالص کوارک‌های `u`، `d`، `s`، `c` و `b` به دست می‌آید. کد ۱٬۲۸۵ ویژگی می‌سازد که ویژگی‌های مشتق‌شده از فضای فاز و توازن اعداد کوانتومی را هم شامل می‌شوند.

**بررسی حضور ذره در پیش‌پردازش** — کد فقط بررسی می‌کند که نام ذره وارد شده باشد تا مشخص کند کدام جایگاه پر است.

**ستون‌های هدف، گروه‌بندی و بررسی داده**

- `Decay Value` مقدار هدف آموزش است و برای پیدا کردن رکورد متناظر در فایل منبع هم استفاده می‌شود.
- `Decay Description` برای پیدا کردن رکورد منبع و گزارش‌دهی استفاده می‌شود.
- `Decay Value Error` برای بررسی عدم‌قطعیت‌ها استفاده می‌شود.

### `All PDG Decays 2026.csv`

```text
PDG ID
Decay Description
Decay Value
```

اسکریپت با تطبیق `Decay Description` و `Decay Value`، شناسهٔ `PDG ID` هر ردیف آموزشی را پیدا می‌کند و از آن در اعتبارسنجی گروه‌بندی‌شده استفاده می‌کند.

## روش انتخاب دیاگرام برای هر ردیف

1. از میان جایگاه‌های ۱، ۲ و ۳، فقط دیاگرام‌هایی بررسی می‌شوند که مقدار `Topology` دارند.
2. از میان این جایگاه‌ها، دیاگرامی انتخاب می‌شود که بزرگ‌ترین مقدار عددی و متناهی `Relative Amplitude` را دارد. در صورت برابری، جایگاه با شمارهٔ کوچک‌تر انتخاب می‌شود. برای نمونه، اگر هر سه توپولوژی موجود باشند و دامنه‌ها به‌ترتیب `0.20`، `0.65` و `0.15` باشند، کد دیاگرام ۲ را انتخاب می‌کند.
3. اگر هیچ‌یک از دیاگرام‌های دارای توپولوژی، دامنهٔ عددی و متناهی نداشته باشد، نخستین جایگاه دارای توپولوژی انتخاب می‌شود. اگر همهٔ توپولوژی‌ها خالی باشند، ویژگی‌های دیاگرام به‌صورت نامشخص ثبت می‌شوند.
4. کد از توپولوژی مرحلهٔ اول، نخستین گذار کوارکی و عنصر مؤثر ماتریس CKM ثبت‌شده برای دیاگرام منتخب استفاده می‌کند. دامنهٔ نسبی فقط برای رتبه‌بندی دیاگرام‌ها به کار می‌رود.

## اعتبارسنجی

امتیازهای گزارش‌شده از پنج بخش بیرونی با گروه‌بندی بر اساس منبع به دست آمده‌اند. هر جست‌وجوی Optuna از سه بخش داخلی گروه‌بندی‌شده استفاده می‌کند. پارامترهای مقیاس‌بندی مقدار هدف فقط از داده‌های بخش آموزش محاسبه می‌شوند. کانال‌های مرتبط با شناسه‌های منبع متفاوت و واپاشی‌های یک نوع ذرهٔ اولیه ممکن است در بخش‌های مختلف قرار بگیرند. ردیف‌های علامت‌گذاری‌شده در ارزیابی لحاظ شده‌اند و نتایج به همین روش تقسیم‌بندی مربوط‌اند.
