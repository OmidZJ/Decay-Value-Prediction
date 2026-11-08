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

هر دو مدل از ۴۰ آزمون Optuna در هر بخش بیرونی، همان ۱٬۲۸۵ ورودی مدل و `huber_slope = 0.8` استفاده کردند. گروه B شامل واپاشی مزون‌ها و باریون‌های دارای محتوای خالص باتمِ غیرصفر است و هر ۱۲۰ ردیف علامت‌گذاری‌شدهٔ آن حفظ شد.

#### روش تقسیم‌بندی داده‌ها به فولد

داده‌ها به پنج بخش بیرونی (فولد) تقسیم شدند. ردیف‌هایی که به یک گروه منبع در PDG تعلق دارند، در یک بخش قرار گرفتند. هر بار یک بخش برای آزمون کنار گذاشته شد و مدل از ابتدا روی چهار بخش دیگر آموزش دید. Optuna تنظیمات مدل را با سه بخش داخلی که از همان داده‌های آموزشی ساخته شدند، انتخاب کرد.

برای مدل مخصوص B، از همان تقسیم‌بندی استفاده کردیم و در هر بخش فقط ردیف‌های B را نگه داشتیم. بنابراین، در هر مرحله ردیف‌های آموزشی B و ردیف‌های آزمون B برای هر دو مدل یکسان بودند. امتیاز نهایی با کنار هم گذاشتن پیش‌بینی‌های آزمون در هر پنج بخش محاسبه شد و هر ردیف ارزیابی‌شده یک پیش‌بینی داشت.

| مدل و نوع امتیاز | نمونهٔ بخش اول | ردیف‌های ارزیابی در مجموع پنج بخش | $R^2$ | RMSE (dex) | MAE (dex) |
| --- | --- | --- | ---: | ---: | ---: |
| مدل مخصوص B، امتیاز B | آموزش روی ۶۰۶ ردیف B<br>ارزیابی روی ۱۴۵ ردیف B | ۷۵۱ ردیف B | 0.74609 | 0.63591 | 0.46307 |
| مدل همهٔ ذرات، امتیاز کلی | آموزش روی ۶۰۶ ردیف B و ۲٬۱۵۳ ردیف دیگر<br>ارزیابی روی همهٔ ۶۹۰ ردیف کنارگذاشته‌شده | همهٔ ۳٬۴۴۹ ردیف | 0.80135 | 0.69582 | 0.47454 |
| مدل همهٔ ذرات، امتیاز B | همان مدل آموزش‌دیدهٔ ردیف بالا<br>ارزیابی روی همان ۱۴۵ ردیف B که مدل مخصوص B روی آن‌ها ارزیابی شد | همان ۷۵۱ ردیف B | **0.75530** | **0.62427** | **0.45879** |

#### گزارش نتایج در سه حالت

1. **آموزش فقط با واپاشی‌های B و ارزیابی روی B.** مقدار $R^2$ روی ۷۵۱ ردیف آزمون B برابر با **0.74609** بود.
2. **آموزش و ارزیابی روی واپاشی‌های همهٔ ذرات.** مقدار $R^2$ روی مجموع ۳٬۴۴۹ ردیف آزمون برابر با **0.80135** بود.
3. **آموزش با واپاشی‌های همهٔ ذرات و ارزیابی فقط روی B.** از پیش‌بینی‌های همان مدل حالت دوم استفاده کردیم و امتیاز را فقط روی همان ۷۵۱ ردیف B در حالت اول حساب کردیم. مقدار $R^2$ برابر با **0.75530** بود.

اضافه کردن واپاشی‌های ذرات دیگر به داده‌های آموزش، مقدار $R^2$ را روی همان ردیف‌های B به اندازهٔ ۰٫۰۰۹۲۱ افزایش داد. اجرای مخصوص B با احتساب جست‌وجوی جداگانه برای ساخت مدل نهایی ۱۲٫۲ دقیقه طول کشید.

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
