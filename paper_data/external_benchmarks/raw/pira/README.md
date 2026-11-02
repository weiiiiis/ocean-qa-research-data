---
license: cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
- config_name: mcqa
  data_files:
  - split: train
    path: mcqa/train-*
  - split: validation
    path: mcqa/validation-*
  - split: test
    path: mcqa/test-*
- config_name: paraphrases
  data_files:
  - split: train
    path: paraphrases/train-*
- config_name: pira_version1
  data_files:
  - split: train
    path: pira_version1/train-*
dataset_info:
- config_name: default
  features:
  - name: id_qa
    dtype: string
  - name: corpus
    dtype: int64
  - name: question_en_origin
    dtype: string
  - name: question_pt_origin
    dtype: string
  - name: question_en_paraphase
    dtype: string
  - name: question_pt_paraphase
    dtype: string
  - name: answer_en_origin
    dtype: string
  - name: answer_pt_origin
    dtype: string
  - name: answer_en_validate
    dtype: string
  - name: answer_pt_validate
    dtype: string
  - name: abstract
    dtype: string
  - name: eid_article_scopus
    dtype: string
  - name: question_generic
    dtype: float64
  - name: answer_in_text
    dtype: float64
  - name: answer_difficulty
    dtype: float64
  - name: question_meaningful
    dtype: float64
  - name: answer_equivalent
    dtype: float64
  - name: question_type
    dtype: string
  - name: abstract_translated_pt
    dtype: string
  - name: pt_question_translated_to_en
    dtype: string
  - name: at_labels
    dtype: float64
  splits:
  - name: train
    num_bytes: 8002269
    num_examples: 1806
  - name: validation
    num_bytes: 994524
    num_examples: 225
  - name: test
    num_bytes: 940555
    num_examples: 227
  download_size: 3976683
  dataset_size: 9937348
- config_name: mcqa
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: question
    dtype: string
  - name: A
    dtype: string
  - name: B
    dtype: string
  - name: C
    dtype: string
  - name: D
    dtype: string
  - name: E
    dtype: string
  - name: correct
    dtype: string
  - name: alternative
    dtype: string
  splits:
  - name: train
    num_bytes: 4327619
    num_examples: 1798
  - name: validation
    num_bytes: 582526
    num_examples: 225
  - name: test
    num_bytes: 551723
    num_examples: 227
  download_size: 2148096
  dataset_size: 5461868
- config_name: paraphrases
  features:
  - name: question_AUT_EN_1
    dtype: string
  - name: question_AUT_EN_2
    dtype: string
  - name: answer_AUT_EN_1
    dtype: string
  - name: answer_AUT_EN_2
    dtype: string
  - name: question_AUT_PT_1
    dtype: string
  - name: question_AUT_PT_2
    dtype: string
  - name: answer_AUT_PT_1
    dtype: string
  - name: answer_AUT_PT_2
    dtype: string
  splits:
  - name: train
    num_bytes: 1175020
    num_examples: 1806
  download_size: 720519
  dataset_size: 1175020
- config_name: pira_version1
  features:
  - name: id_qa
    dtype: string
  - name: corpus
    dtype: int64
  - name: question_en_origin
    dtype: string
  - name: question_pt_origin
    dtype: string
  - name: question_en_paraphase
    dtype: string
  - name: question_pt_paraphase
    dtype: string
  - name: answer_en_origin
    dtype: string
  - name: answer_pt_origin
    dtype: string
  - name: answer_en_validate
    dtype: string
  - name: answer_pt_validate
    dtype: string
  - name: eid_article_scopus
    dtype: string
  - name: text_excerpts_un_reports
    dtype: string
  - name: question_generic
    dtype: bool
  - name: answer_in_text
    dtype: bool
  - name: answer_difficulty
    dtype: float64
  - name: question_meaningful
    dtype: float64
  - name: answer_equivalent
    dtype: float64
  - name: question_type
    dtype: string
  splits:
  - name: train
    num_bytes: 3096316
    num_examples: 2271
  download_size: 1342133
  dataset_size: 3096316
task_categories:
- question-answering
language:
- pt
- en
tags:
- climate
size_categories:
- 1K<n<10K
---

# Pirá: A Bilingual Portuguese-English Dataset for Question-Answering about the Ocean, the Brazilian coast, and climate change

Pirá is a crowdsourced reading comprehension dataset on the ocean, the Brazilian coast, and climate change. 
QA sets are presented in both Portuguese and English, together with their corresponding textual context.
The dataset also contains human and automatic paraphrases for questions and answers, as well as a number of qualitative assessments. 

The original paper was published at CIKM'21 and can be found [here](https://dl.acm.org/doi/pdf/10.1145/3459637.3482012). 
As a subsequent project, we have produced a curated version of the dataset, which we refer to as Pirá 2.0. 
In this step, we have also defined a number of benchmarks and reported the corresponding baselines.
This is the version that we make available at HuggingFace. 
Pirá 2.0's preprint is available in [Arxiv](https://arxiv.org/abs/2309.10945). 

Pirá is, to the best of our knowledge, the first QA dataset with supporting texts in Portuguese, and, perhaps more importantly, 
the first bilingual QA dataset that includes Portuguese as one of its languages. 
Pirá is also the first QA dataset in Portuguese with unanswerable questions so as to allow the study of answer triggering. 
Finally, it is the first QA dataset that tackles scientific knowledge about the ocean, climate change, and marine biodiversity.

More information on the methodology, dataset versions, and benchmarks can be found on the project's [Github page](https://github.com/C4AI/Pira/).
You can also find there the Multiple-Choice version of Pirá.

# Dataset
The dataset is split into train, validation, and test sets. 

| Split | Size | #QAs |
|---|---|---|
| Training | 80% | 1806 | 
| Validation | 10% | 225 |
| Test | 10% | 227 |
| Full dataset | 100% | 2258 |

Above is an example of a question-answer set from Pirá:

```
{
    'id_qa': 'B2142',
    'corpus": 2,
    'question_en_origin': 'What are the proportion of men and women employed in the fishery sector worlwide?',
    'question_pt_origin': 'Qual é a proporção de homens e mulheres empregados no setor pesqueiro em todo o mundo?',
    'question_en_paraphase': 'Which share of the fishery sector workers of the world are women?',
    'question_pt_paraphase': 'Qual parcela dos trabalhadores do setor da pesca no mundo são mulheres?',
    'answer_en_origin': '85 per cent men and 15 per cent women.',
    'answer_pt_origin': '85 por cento homens e 15 por cento mulheres.',
    'answer_en_validate': 'It is estimated that more than fifteen per cent of the fishing sector workers are women.',
    'answer_pt_validate': 'Estima-se que mais de quinze por cento dos trabalhadores do setor da pesca são mulheres.',
    'eid_article_scopus': '',
    'text_excerpts_un_reports': 'Distribution of ocean benefits and disbenefits Developments in employment and income from fisheries and aquaculture The global harvest of marine capture fisheries has expanded rapidly since the early 1950s and is currently estimated to be about 80 million tons a year. That harvest is estimated to have a first (gross) value on the order of 113 billion dollars. Although it is difficult to produce accurate employment statistics, estimates using a fairly narrow definition of employment have put the figure of those employed in fisheries and aquaculture at 58.3 million people (4.4 per cent of the estimated total of economically active people), of which 84 per cent are in Asia and 10 per cent in Africa. Women are estimated to account for more than 15 per cent of people employed in the fishery sector. Other estimates, probably taking into account a wider definition of employment, suggest that capture fisheries provide direct and indirect employment for at least 120 million persons worldwide. Small-scale fisheries employ more than 90 per cent of the world’s capture fishermen and fish workers, about half of whom are women. When all dependants of those taking full- or part-time employment in the full value chain and support industries (boatbuilding, gear construction, etc.) of fisheries and aquaculture are included, one estimate concludes that between 660 and 820 million persons have some economic or livelihood dependence on fish capture and culture and the subsequent direct value chain. No sound information appears to be available on the levels of death and injury of those engaged in capture fishing or aquaculture, but capture fishing is commonly characterized as a dangerous occupation. Over time, a striking shift has occurred in the operation and location of capture fisheries. In the 1950s, capture fisheries were largely undertaken by developed fishing States. Since then, developing countries have increased their share. As a broad illustration, in the 1950s, the southern hemisphere accounted for no more than 8 per cent of landed values. By the last decade, the southern hemisphere’s share had risen to 20 per cent. In 2012, international trade represented 37 per cent of the total fish production in value, with a total export value of 129 billion dollars, of which 70 billion dollars (58 per cent) was exports by developing countries. Aquaculture is responsible for the bulk of the production of seaweeds. Worldwide, reports show that 24.9 million tons was produced in 2012, valued at about 6 billion dollars. In addition, about 1 million tons of wild seaweed were harvested. Few data were found on international trade in seaweeds, but their culture is concentrated in countries where consumption of seaweeds is high.',
    'question_generic': false,
    'answer_in_text': true,
    'answer_difficulty': 1,
    'question_meaningful': 5,
    'answer_equivalent': 5,
    'question_type': 'None of the above'
  }
  ```


# Automatic Paraphrases
As we have only generated automatic paraphrases for questions and answers in the train set, they had to be saved in a different Dataset file.

To download the automatic paraphrases, just run:

```
paraphrases = load_dataset("paulopirozelli/pira", "paraphrases")
```

# Multiple Choice Question Answering
We have also developed a multiple choice question answering version of Pirá 2.0.

To download the automatic paraphrases, just run:

```
mcqa = load_dataset("paulopirozelli/pira", "mcqa")
```

Above is an example of a question-answer set from Pirá:

```
{
  'id_qa': 'A1582',
   'corpus': 1,
   'question_en_origin': 'In the estuary, with marine influence, what was associated to deep areas with sandy sediment?',
   'question_pt_origin': 'No estuário, com influência marinha, o que foi associado a áreas profundas com sedimento arenoso?',
   'question_en_paraphase': 'What was discovered in estuary under deep areas with sand sediment and marine influence?',
   'question_pt_paraphase': 'O que foi descoberto no estuário sob áreas profundas com sedimento arenoso e influência marítima?',
   'answer_en_origin': 'The Laryngosigma lactea and Pyrgo oblonga foraminifera species.',
   'answer_pt_origin': 'As espécies Laryngosigma lactea e Pyrgo oblonga de foraminíferos.',
   'answer_en_validate': 'The species Laryngosigma lactea and Pyrgo oblonga.',
   'answer_pt_validate': 'A espécie Laryngosigma lactea e Pyrgo oblonga.',
   'eid_article_scopus': '2-s2.0-85092100205',
   'text_excerpts_un_reports': None,
   'question_generic': False,
   'answer_in_text': True,
   'answer_difficulty': 4.0,
   'question_meaningful': 5.0,
   'answer_equivalent': 4.0,
   'question_type': 'Who'
}
```


# Pirá 1.0
You can also access the original Pirá dataset. Just run:

```
pira1 = load_dataset("paulopirozelli/pira", "pira_version1")
```