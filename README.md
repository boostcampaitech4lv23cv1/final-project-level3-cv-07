# CAFE: CArtoonize For Extra faces

## Member
<table>
    <th colspan=5>블랙박스</th>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/kimk-ki"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/110472164?v=4"/></a>
            <br />
            <a href="https://github.com/kimk-ki"><strong>🙈 김기용</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/SeongSuKim95"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/62092317?v=4"/></a>
            <br/>
            <a href="https://github.com/SeongSuKim95"><strong>🐒 김성수</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/juye-ops"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/103459155?v=4"/></a>
            <br/>
            <a href="https://github.com/juye-ops"><strong>🙉 김주엽</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/99sphere"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/59161083?v=4"/></a>
            <br />
            <a href="https://github.com/99sphere"><strong>🙊 이  구</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/thlee00"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/56151577?v=4"/></a>
            <br/>
            <a href="https://github.com/thlee00"><strong>🐵 이태희</strong></a>
            <br />
        </td>
    </tr>
</table>

- 김기용_T4020: Cartoonize 모델 조사, 실험 결과 분석
- 김성수_T4039: Object Tracking 모델 조사, Modeling, 알고리즘 개발
- 김주엽_T4048: Model Serving
- 이  구_T4145: Cartoonize 모델 조사, Modeling, 알고리즘 개발
- 이태희_T4172: Object Tracking 모델 조사, Modeling, 코드 오류 분석 및 수정

## 프로젝트 개요

<table>
    <tr>
        <td align="center">
            <a><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216925652-aa5b48f9-9225-4d1f-8f90-d252b0ecce46.png"/></a>
            <br />
        </td>
        <td align="center">
            <a><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216926107-e4413d9c-eee4-4722-85bc-f1bef7ceef00.png"/></a>
            <br/>
        </td>
    </tr>
</table>

- TV 프로그램 혹은 유튜브를 보다보면, 메인 출연자가 아닌 이들의 얼굴을 모자이크 된 것을 쉽게 찾아볼 수 있다. 하지만 이러한 모자이크 기법은 얼굴의 특징을 지워버리기 때문에 인물의 얼굴 표정, 눈빛, 시선과 같은 정보를 잃게 된다. 

<table>
    <tr>
        <td align="center">
            <a><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216928691-458d84a1-37be-4690-93d2-52106c94c898.png"/></a>
            <br />
        </td>
        <td align="center">
            <a><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216928817-dffec866-bafe-49f4-8c89-9e82621d7807.png"/></a>
            <br/>
        </td>
    </tr>
</table>

- 위의 사진들은 '백종원의 쿠킹로그'라는 유튜브 채널에서 가져온 것으로, 일반인의 표정과 대응되는 백종원의 사진으로 대체함으로써 그들의 반응을 효과적으로 파악할 수 있도록 하였다. 
- 하지만, 이러한 방식은 편집자가 직접 해당 프레임의 얼굴을 찾아 바꿔주어야 하기 때문에 상당한 비용(시간, 노력 등)이 발생한다.
- 이에 우리는 기존의 모자이크를 대체하여 사람을 특정할 수 있을 정도로 얼굴을 노출시키지 않는 동시에 얼굴 표정, 시선, 눈빛과 같은 정보는 보존할 수 있는 새로운 방식, `CAFE(CArtoonize For Extra faces)`를 제안한다.

## 프로젝트 환경
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Architecture
### Model Flow
<table>
        <td align="center">
            <a><img height="500px" width="800px" src="https://user-images.githubusercontent.com/59161083/216927654-a4676796-80a2-4802-bdba-97449debbf39.png"/></a>
            <br/>
        </td>
</table>

전체적인 Model Flow는 아래와 같다.    

> 1. User로부터 영상과 영상의 주인공(target) 사진을 입력받는다.   
> 2. 영상에 face detection & tracking, cartoonize를 적용한다.     
    2-1. 영상에 대한 face detection, tracking을 통하여 모든 등장인물의 얼굴 정보를 얻는다.    
    2-2. 영상의 모든 프레임에 대한 cartoonize를 진행한다.
> 3. 주인공 사진과 영상에 등장하는 인물들의 사진에 대한 feature를 뽑아낸 후, cosine similarity를 계산하여 target과 target이 아닌 얼굴들을 구분한다. 
> 4. target이 아닌 얼굴들에 대한 정보(from 2-1)를 이용하여, 모든 프레임의 얼굴을 swap 한다. 

### Service Flow
<table>
        <td align="center">
            <a><img height="500px" width="1000px" src="https://user-images.githubusercontent.com/59161083/216934011-ef53c2f2-f144-4e21-a465-a65935dcd0b5.png"/></a>
            <br/>
        </td>
</table>

전체적인 Service Flow는 아래와 같다.    
> 1. Streamlit을 통해 user와 interaction하며, 주인공 이미지와 영상을 입력받고, 결과물을 다운받을 수 있다.    
> 2. Streamlit을 통해 입력받은 이미지와 영상은 local file storage에 저장되며, FastAPI를 통해 Detection & Tracking, Cartoonize 연산을 요청한다.
> 3. Detection & Tracking은 PyTorch 환경에서 실행되고, Cartoonize는 Tensorflow 환경에서 실행된다. 이 과정은 병렬적으로 진행되며, Tracking 결과는 MongoDB에 저장된다. 
> 4. 위의 과정이 끝난 이후, backend에서 MongoDB에 저장된 tracking 정보를 사용하여 face swapping 과정을 수행한다.
> 5. Streamlit을 통해 user가 최종 결과물의 재생 및 저장이 가능하다. 


## Usage
```(python)
# Clone our repository
git clone https://github.com/boostcampaitech4lv23cv1/final-project-level3-cv-07

# Move to our dir
cd final-project-level3-cv-07

# Setup for each virtual environment (for frontend, backend, detection & tracking, cartoonize)
bash init.sh

# Open frontend/app.py and update backend ip address in line 12
vim frontend/app.py

"""
fix below line
backend = "http://115.85.182.51:30002" -> backend = {your_backend_ip_address}
"""

# Start all process required for starting demo page
bash start.sh
```

## Demo


## Wrap-Up Report

## Reference