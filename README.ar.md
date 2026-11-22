# JAIT — تشغيل نماذج TAIT في المتصفح

JAIT هو محرك TypeScript يشغّل نماذج `.npz` المدرَّبة بـ TAIT مباشرة في
المتصفح. بلا خادم، بلا حساب، ولا تغادر أي بيانات الجهاز: يُنزَّل النموذج
مباشرة من Hugging Face (أو يُفتح من ملف المستخدم نفسه) ويُنفَّذ محليا.

English: [README.md](README.md) · 简体中文: [README.zh-CN.md](README.zh-CN.md)

ملف `.npz` واحد، ثلاثة مشغلات:

- **TAIT** يدرّب (Python / NumPy، على الهاتف) —
  [muzammilsoft/tait](https://github.com/muzammilsoft/tait)
- **KAIT** يشغّل على الهاتف (Kotlin) —
  [muzammilsoft/KAIT-Studio](https://github.com/muzammilsoft/KAIT-Studio)
- **JAIT** يشغّل في المتصفح (TypeScript) — هذا المستودع

## الأداء

`forward()` لا تخصّص شيئا: كل مصفوفة وسيطة تعيش في تخزين مؤقت خاص
بكل نموذج يُعاد استخدامه عبر الاستدعاءات، فلا ينتج الرد أي قمامة
لجامع القمامة (قبل الإصلاح، كانت تمريرة واحدة تخصّص 12 مصفوفة
`Float64Array` — نحو 0.5 ميغابايت قمامة لكل رد قصير؛ الآن صفر
للتمريرة). الاحتمالات المُرجَعة هي عرض في ذلك التخزين المؤقت:
استهلكها أو انسخها قبل استدعاء `forward()` التالي على النموذج نفسه.

## ما تم التحقق منه

بوابة المرحلة 1 — `npm test` يحمّل TinyBili
(`muzammilsoft/tinybili`، `tinyGPT0.1.npz`، 1,072,206 بايت) ويقارن
مع Python TAIT 2.3.0:

- معرفات المُرمِّز مطابقة تماما للموجهات العربية والإنجليزية
- أعلى 5 رموز تالية مطابقة، وأقصى فرق في الاحتمال **2.22e-16**
  (البوابة: أقل من 1e-3)
- فك التشفير الجشع مطابق لـ Python
- ملف غير TAIT يُرفض بخطأ واضح

يُحمَّل الاختبار من Hugging Face عند غياب نسخة محلية
(`JAIT_TEST_NPZ=/path/to/model.npz` يتجاوز المصدر).
القيم المرجعية في `test/reference.json`، مولَّدة من مصدر TAIT الحقيقي
في `workspace/tait-dev`.

## صيغة النموذج (ما تنفذه JAIT)

منقول سطرا بسطر من `tait/core/model.py` و `tait/core/tokenizer.py`:

- **NPZ**: ملف ZIP من مصفوفات `.npy` (STORED أو DEFLATE). الأوزان:
  `wte, wpe, Wq, Wk, Wv, Wo, W1, b1, W2, b2, Whead, bhead`، إضافة إلى
  بيانات BPE `merge_a, merge_b, merge_id` والقياسات `sep_id, end_id,
  tok_pad_id, block_size, pad_id`. `wte`/`wpe` من نوع float32، والباقي
  float64؛ والدمجات int32.
- **المُرمِّز**: BPE على مستوى البايت. المفردات هي 256 بايتا مفردا +
  الدمجات + 3 رموز خاصة (SEP 1000، END 1001، PAD 1002 في TinyBili).
  الترميز يدمج الزوج ذا أقل معرف دمج أولا. ملف NPZ مكتفٍ بذاته: لا
  حاجة أبدا إلى `vocab.json`.
- **النموذج**: كتلة محوِّل واحدة **بلا LayerNorm** (TAIT لا يستخدمها).
  انتباه ذاتي سببي مع قناع `-1e9`، وMLP بـ ReLU، واتصالات متبقية.
  جمع التضمين يُقرَّب بـ `Math.fround` لأن NumPy يجمع تضمينات float32
  في float32.
- **التوليد**: نفس الترتيب كما في Python — تصفير PAD، قوة درجة
  الحرارة، top-k، top-p، ثم سحب واحد. TinyBili نموذج تجريبي صغير
  (6,666 مثالا، 50 حقبة)، فإجاباته غالبا مشوشة؛ JAIT تعرضه بصراحة
  كنموذج تجريبي.

## فرق صريح عن Python

Python يسحب باستخدام مولد PCG64 في NumPy؛ JavaScript لا مثيل لها.
JAIT تستخدم mulberry32، فالبذرة نفسها تعطي عينة *مختلفة لكن قابلة
للتكرار* في JS عنها في Python. المقارنة الحتمية يجب أن تستخدم أعلى 5
احتمالات أو فك التشفير الجشع، لا بذرة مسحوبة أبدا.

## الاستخدام

التثبيت (JAIT ليست على npm بعد — ثبّتها من GitHub؛ البناء يعمل
تلقائيا عند التثبيت):

```sh
npm install github:muzammilsoft/jait
```

أو استنسخ وابنِ بنفسك:

```sh
git clone https://github.com/muzammilsoft/jait
cd jait
npm install
npm run build   # -> dist/
npm test        # build + Phase 1 gate
```

ثم:

```ts
import { loadModelFromUrl, generateReply } from 'jait';

const { model, tokenizer } = await loadModelFromUrl(
  'https://huggingface.co/muzammilsoft/tinybili/resolve/main/tinyGPT0.1.npz',
);
const reply = generateReply(model, tokenizer, 'مرحبا', {
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  maxNewTokens: 64,
});
```

أو حمّل من ملف المستخدم نفسه:

```ts
import { loadModelFromNpz } from 'jait';
const bytes = new Uint8Array(await file.arrayBuffer());
const { model, tokenizer } = loadModelFromNpz(bytes);
```

اعتماد التشغيل الوحيد: `fflate` (MIT) لـ ZIP/DEFLATE. TypeScript أداة
تطوير فقط؛ الناتج JavaScript عادي.

JAIT مشروع مفتوح المصدر مستقل من Khartoum Ghoul. ليست تابعة لـ
Bilibili أو Hugging Face أو أي مضيف نماذج.
