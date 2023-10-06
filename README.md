## EconomicCycle

2023학년도 2학기 경기변동론(캡스톤디자인) 프로젝트 페이지입니다. 대한민국 경기국면 예측을 위한 Double Machine Learning / eXplainable AI model을 구축하는 것을 목표로 하고 있습니다.

### Project Blueprint

![blueprint.png](Image%2Fblueprint.png)

- 후행지표 : Stock Market, Bond Market, Unemployment
- 선행지표 : Investment, Export
- 동행지표 : Inflation, Interest Rate

#### Primary Modeling

각각의 지표를 예측하는 모형을 제작합니다. Machine Learning 모형을 각각 구축하면 class에 대한 추정 확률이 계산됩니다. 이를 Secondary Model을 위한 input으로 사용합니다.

#### Secondary Modeling

Primary Model에서 추정된 확률을 Feature로 가공하여 input을 넣습니다. output은 class가 두 개 혹은 네 개인 분류 모형으로, State Space로 작용할 수 있습니다.

### Primary Models

#### Bond Market

![bond_spread.png](Image%2Fbond_spread.png)

케인즈의 유동성선호가설에 기반한 모형을 고려하였습니다. Gaussian Mixtual Model을 사용하였습니다.

#### Stock Market

![stock_market.png](Image%2Fstock_market.png)

### Double Machine Learning 모형의 장점

기존의 Machine Learning 모형은 Ensemble Learning이 아니면 어떠한 Factor가 영향을 끼치는지 분석하기가 까다로웠습니다. Double Machine Learning 모형을 사용하면, 모형별로 추론이 가능해지기 때문에, 기존 회귀분석처럼 Treatment효과를 분석할 수 있게 됩니다. 
이는 차후 경제학 연구에서 중요히 쓰일 모형이 될 것으로 보입니다.