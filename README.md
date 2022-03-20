# korean relation extraction (korre)
한국어 문장 내에 존재하는 개체 사이의 관계를 추출하는 모듈입니다.

## Quick Start
먼저 `korre` 패키지를 다음과 같이 import 합니다.
```python
>>> from korre import KorRE
```

패키지를 import 한 후에는, 다음과 같이 관계 추출을 수행할 수 있는 객체를 만들어줍니다. 객체가 만들어지면서, 학습된 모델을 자동으로 로드합니다.
```python
>>> from korre import KorRE
>>> korre = KorRE()
```

## Named Entity Recognition
문장 내에 내포되어 있는 관계를 추출하기 위해 개체명 인식이 필요합니다.
한국어 개체명 인식 모듈로는 `pororo`(https://github.com/kakaobrain/pororo) 모듈을 그대로 사용하였고 `korre`에서는 다음과 같이 사용 가능합니다.
