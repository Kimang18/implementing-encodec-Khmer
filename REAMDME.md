# EnCodec: the beginning of Speech Tokenization

EnCodec ជាម៉ូដែលសំរាប់បង្ហាប់ទិន្នន័យសំលេង(Neural Audio Compression model) អភិវឌ្ឍន៍ដោយក្រុមហ៊ុន Meta ហើយត្រូវចេញផ្សាយក្នុងអត្ថបទស្រាវជ្រាវ [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)។
បច្ចុប្បន្ន EnCodec បានក្លាយជាម៉ូដែលគ្រឹះសំរាប់ការបង្កើតម៉ូដែល សំលេងទៅសំលេង (Speech-to-Speech) ដោយ EnCodec ផ្តល់ឳកាសអោយអ្នកស្រាវជ្រាវធ្វើ Tokenization ទៅលើសំលេង។
នេះជាដំណោះស្រាយដ៏សំខាន់ ពីព្រោះថាការធ្វើ Tokenization លើសំលេង អនុញ្ញាតអោយយើងបង្ហាត់ AI អោយចេះស្តាប់ និងឆ្លើយឆ្លងជាមួយក្នុងទំរង់ជាសំលេងតែម្តង ដោយមិនចាំបាច់ឆ្លងកាត់ការឆ្លើយក្នុងទំរង់ជាអត្ថបទដូច ChatGPT ធ្លាប់ធ្វើកន្លងមកនោះឡើយ។

# Training EnCodec on your Dataset

អ្នកទាំងអស់គ្នាអាចបង្ហាត់ EnCodec លើផ្ទាល់ខ្លួនបាន ដោយប្រើប្រាស់កូដមានស្រាប់ក្នុង repo នេះ។
បច្ចុប្បន្ន យើងបង្ហាត់ម៉ូដែលលើភាសាខ្មែរ ដោយមិនចាំបាច់ប្តូរទាំងអស់ ដោយគ្រាន់ក្លូនកូដពី repo នេះ រួចបញ្ចូលកូដខាងក្រោមក្នុង terminal
```bash
python train.py
```

សំរាប់ការផ្លាស់ប្តូរ configuration សូមចូលទៅមើលfile `configs/config.yaml`។
អ្នកអាចប្តូរតំលៃជម្រើសក្នុង `config.yaml` តាមរយៈ terminal បាន។
ឧទាហរណ៍ បើអ្នកចង់ប្តូរ batch_size មកស្មើ 8 សូមបញ្ចូលកូដខាងក្រោមក្នុង terminal
```bash
python train.py ++data.bs_per_device=8
```

បើអ្នកមានទិន្នន័យលើ Huggingface ហើយចង់ប្រើប្រាស់វាដើម្បីបង្ហាត់ សូមអ្នកបញ្ចូលកូដខាងក្រោមក្នុង terminal
```bash
python train.py ++data.bs_per_device=8 ++data.hf_repo="huggingface_repo_to_your_data"
```

# Pytorch Lightning

កូដដែលខ្ញុំបានសរសេរនេះប្រើប្រាស់ Pytorch Lightning ដែលជាសំរួល boilerplate មួយចំនួន។
ក្នុងនោះ អ្នកទាំងអស់គ្នាអាច montior training បានដោយប្រើប្រាស់ `tensorboard`។
យើងអាចមើល logging ក្នុង `tensorboard` ដោយបញ្ចូលកូដខាងក្រោមក្នុង terminal មួយទៀត
```bash
tensorboard --logdir lightning_logs/
```