# KORean Relation Extraction (korre)
한국어 문장 내에 존재하는 개체 사이의 관계를 추출하는 모듈입니다.

한국어 기반의 관계 추출 모델을 학습하기 위해 한국어를 기반으로 사전학습된 BERT 모델을 이용하였습니다.

BERT pretrained model로는 [KR-BERT-MEDIUM](https://github.com/snunlp/KR-BERT-MEDIUM)을 활용하였습니다. 


## Installation
- `korre` 는 `python>=3.8` 환경을 필요로 합니다.
- 다음과 같이 모듈을 설치하고 추가적으로 필요한 라이브러리를 설치합니다.
```console
git clone https://github.com/datawhales/korre.git
cd korre
pip install -r requirements.txt
```


## Quick Start
먼저 `korre` 패키지를 다음과 같이 import 합니다.
```python
>>> from korre import KorRE
```


패키지를 import 한 후에는, 다음과 같이 관계 추출을 수행할 수 있는 객체를 만들어줍니다. 

객체가 만들어지면서, 학습된 모델을 자동으로 로드합니다.
```python
>>> korre = KorRE()
```


## Named Entity Recognition
문장 내에 내포되어 있는 관계를 추출하기 위해 개체명 인식이 필요합니다. 

한국어 개체명 인식 모듈로는 `pororo`(https://github.com/kakaobrain/pororo) 모듈을 그대로 사용하였고 `korre`에서는 다음과 같이 사용 가능합니다.
```python
>>> korre.pororo_ner('갤럭시 플립2는 삼성에서 만든 스마트폰이다.')
[('갤럭시 플립2', 'ARTIFACT'),
 ('는', 'O'),
 (' ', 'O'),
 ('삼성', 'ORGANIZATION'),
 ('에서', 'O'),
 (' ', 'O'),
 ('만든', 'O'),
 (' ', 'O'),
 ('스마트폰', 'TERM'),
 ('이다.', 'O')]
```
`pororo` 모듈을 통해 개체명 인식을 수행한 후 관계를 추출하기 위한 개체를 추출하여 다음과 같이 문장에서의 인덱스를 함께 나타낼 수 있습니다.

이를 통해 관계를 추출하고자 하는 개체의 인덱스를 입력에 사용하여 관계 추출에 사용하게 됩니다.
```python
>>> korre.ner('갤럭시 플립2는 삼성에서 만든 스마트폰이다.')
[('갤럭시 플립2', 'ARTIFACT', [0, 7]),
 ('삼성', 'ORGANIZATION', [9, 11]),
 ('스마트폰', 'TERM', [17, 21])]
```


## Inference (Relation Extraction)
`korre` 모듈을 통해서는 다음의 3가지 형태로 관계 추출을 수행할 수 있습니다.

- 사용자가 입력한 문장에 관계를 알고자 하는 개체 쌍의 앞뒤에 직접 **entity marker token**을 붙인 경우
- 문장과 관계를 추출하고자 하는 두 개체의 위치 인덱스를 직접 입력하는 경우
- 문장만 입력하여 내포되어 있는 모든 관계를 알고자 하는 경우


### 1. **entity marker token**이 존재하는 문장이 입력된 경우
- 입력 예시
```python
>>> korre = KorRE()
>>> korre.infer('[E1] 갤럭시 플립2 [/E1] 는 [E2] 삼성 [/E2] 에서 만든 스마트폰이다.', entity_markers_included=True)
```
- 출력 예시
```python
[('갤럭시 플립2', '삼성', '해당 개체의 제조사(manufacturer)')]
```
**entity marker token**은 개체의 위치를 나타내기 위한 토큰으로, 이를 개체의 앞뒤에 붙인 상태로 `entity_markers_included=True` 옵션을 주게 되면 해당 두 개체 사이에 존재하는 관계를 추출할 수 있습니다.

`korre` 모듈에서는 **entity marker token**으로 개체의 앞에 붙이는 [E1], [E2]와 개체의 뒤에 붙이는 [/E1], [/E2]를 사용합니다.


### 2. 문장과 관계를 추출하고자 하는 두 개체의 위치 인덱스가 입력된 경우
- 입력 예시
```python
>>> korre = KorRE()
>>> korre.infer('갤럭시 플립2는 삼성에서 만든 스마트폰이다.', [0, 7], [9, 11])
```
- 출력 예시
```python
[('갤럭시 플립2', '삼성', '해당 개체의 제조사(manufacturer)')]
```
`korre` 모듈의 메소드 함수 `ner`을 이용하여 문장에 존재하는 개체의 위치 인덱스를 구해 함께 입력한 경우에도 관계 추출을 수행할 수 있습니다.


### 3. 문장만 입력하여 내포되어 있는 모든 관계를 알고자 하는 경우
- 입력 예시
```python
>>> korre = KorRE()
>>> korre.infer('갤럭시 플립2는 삼성에서 만든 스마트폰이다.')
```
- 출력 예시
```python
[('갤럭시 플립2', '삼성', '해당 개체의 제조사(manufacturer)'),
 ('갤럭시 플립2', '스마트폰', '하위 개념(subclass of)'),
 ('삼성', '갤럭시 플립2', '해당 개체의 제품(product or material produced)'),
 ('삼성', '스마트폰', '해당 개체의 제품(product or material produced)'),
 ('스마트폰', '갤럭시 플립2', '해당 개체가 다음으로 이루어져 있음(has part)'),
 ('스마트폰', '삼성', '해당 개체의 제조사(manufacturer)')]
```
개체의 위치가 입력되지 않고 문장만 입력된 경우에는 문장 내에 존재하는 모든 관계를 추출할 수 있습니다.