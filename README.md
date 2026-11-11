# EnCodec: the beginning of Speech Tokenization

EnCodec ជាម៉ូដែលសំរាប់បង្ហាប់ទិន្នន័យសំលេង (Neural Audio Compression model) អភិវឌ្ឍន៍ដោយក្រុមហ៊ុន Meta ហើយត្រូវចេញផ្សាយក្នុងអត្ថបទស្រាវជ្រាវ [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)។
បច្ចុប្បន្ន EnCodec បានក្លាយជាម៉ូដែលគ្រឹះសំរាប់ការបង្កើតម៉ូដែល សំលេងទៅសំលេង (Speech-to-Speech) ដោយ EnCodec ផ្តល់ឳកាសអោយអ្នកស្រាវជ្រាវធ្វើ Tokenization ទៅលើសំលេង។
នេះជាដំណោះស្រាយដ៏សំខាន់ ពីព្រោះថាការធ្វើ Tokenization លើសំលេង អនុញ្ញាតអោយយើងបង្ហាត់ AI អោយចេះស្តាប់ និងឆ្លើយឆ្លងជាមួយយើងក្នុងទំរង់ជាសំលេងតែម្តង ដោយមិនចាំបាច់ឆ្លងកាត់ការឆ្លើយក្នុងទំរង់ជាអត្ថបទដូច ChatGPT ធ្លាប់ធ្វើកន្លងមកនោះឡើយ។

# Training EnCodec on your Dataset

ដំបូង សូមអ្នកទាំងអស់គ្នាចូលទៅមើល file `requirements.txt` ជាមុនសិន ដើម្បីដឹងអំពី dependency។
បើអ្នកទាំងអស់គ្នាទុកចិត្តហើយ អាចតំឡើង dependency តាមរយៈ command
```bash
pip install -r requirments.txt
```

អ្នកទាំងអស់គ្នាអាចបង្ហាត់ EnCodec លើទិន្នន័យផ្ទាល់ខ្លួនបាន ដោយប្រើប្រាស់កូដមានស្រាប់ក្នុង repo នេះ។
បច្ចុប្បន្ន យើងបង្ហាត់ម៉ូដែលលើភាសាខ្មែរ ដោយមិនចាំបាច់ប្តូរអ្វីទាំងអស់ ដោយគ្រាន់ក្លូនកូដពី repo នេះ រួចបញ្ចូលកូដខាងក្រោមក្នុង terminal
```bash
python train.py
```

សំរាប់ការផ្លាស់ប្តូរ configuration សូមចូលទៅមើលfile `configs/config.yaml`។
អ្នកអាចប្តូរតំលៃជម្រើសក្នុង `config.yaml` តាមរយៈ terminal បាន។
ឧទាហរណ៍ បើអ្នកចង់ប្តូរ `batch_size` មកស្មើ 8 សូមបញ្ចូលកូដខាងក្រោមក្នុង terminal
```bash
python train.py ++data.bs_per_device=8
```

បើអ្នកមានទិន្នន័យលើ Huggingface ហើយចង់ប្រើប្រាស់វាដើម្បីបង្ហាត់ សូមអ្នកបញ្ចូលកូដខាងក្រោមក្នុង terminal។

```bash
python train.py ++data.bs_per_device=8 ++data.hf_repo="huggingface_repo_to_your_data"
```

បញ្ជាក់ ទិន្នន័យអ្នកត្រូវតែមាន 2 ចំណែក `train` split និង `test` split ហើយ split និមួយៗត្រូវតែមាន column “audio” ដែលជា column សំរាប់ទាញយកសំលេង។
សូមអ្នករៀបចំទិន្នន័យអោយមានចំណែក 2 នេះ ទើបអាចដំណើរការកូដមានស្រាប់បាន។
បើមិនដូចច្នោះទេ អ្នកត្រូវមានភាពក្លាហានចូលទៅកែកូដមានស្រាប់ 🤣។

# Pytorch Lightning

កូដដែលខ្ញុំបានសរសេរនេះប្រើប្រាស់ Pytorch Lightning ដែលជួយសំរួល boilerplate មួយចំនួនក្នុងបង្ហាត់ AI ក្នុងក្របខណ្ឌ `pytorch` ។
ក្នុងនោះ អ្នកទាំងអស់គ្នាអាច montior training បានដោយប្រើប្រាស់ `tensorboard` (default option)។
យើងអាចមើល logging ក្នុង `tensorboard` ដោយបញ្ចូលកូដខាងក្រោមក្នុង terminal មួយទៀត
```bash
tensorboard --logdir lightning_logs/
```

# Training on Colab

បើមិនទាន់ទុកចិត្តលើកូដរបស់ខ្ញុំ អ្នកអាចសាកបង្ហាត់ដោយប្រើប្រាស់ Google Colab សិនបាន។
សូមចុចទីនេះ ដើម្បីបើក notebook គំរូសំរាប់បង្ហាត់។

# To buy me a coffee

បើអ្នកទាំងអស់ចង់ជួយគាំទ្រការងាររបស់ខ្ញុំ អាចធ្វើការបរិច្ចាគតាម QR Code ខាងក្រោមនេះបាន។ សូមអរគុណ
__Happy Learning__

<figure>
  <div style="text-align: center;"><a name='slotMachine' ><img src="https://kimang18.github.io/assets/fig/aba_qr_kimang.JPG" width="500" /></a></div>
  <figcaption> Using Cambodian bank account, you can donate by scanning my ABA QR code here. (or click <a href="https://pay.ababank.com/oRF8/8yp6hy53">here</a>. Make sure that receiver's name is 'Khun Kim Ang'.) </figcaption>
</figure>