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
    <th colspan=2></th>
    <tr>
        <td align="center">
            <a href="https://github.com/kimk-ki"><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216925652-aa5b48f9-9225-4d1f-8f90-d252b0ecce46.png"/></a>
            <br />
        </td>
        <td align="center">
            <a href="https://github.com/SeongSuKim95"><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216926107-e4413d9c-eee4-4722-85bc-f1bef7ceef00.png"/></a>
            <br/>
        </td>
    </tr>
</table>

TV 프로그램 혹은 유튜브를 보다보면, 메인 출연자가 아닌 이들의 얼굴을 모자이크 된 것을 쉽게 찾아볼 수 있다. 이러한 모자이크 기법은 얼굴의 특징을 지워버리기 때문에, 얼굴 표정, 눈빛, 시선과 같은 정보를 잃게 된다. 

<table>
    <th colspan=2></th>
    <tr>
        <td align="center">
            <a href="https://github.com/kimk-ki"><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216928691-458d84a1-37be-4690-93d2-52106c94c898.png"/></a>
            <br />
        </td>
        <td align="center">
            <a href="https://github.com/SeongSuKim95"><img height="300px" width="500px" src="https://user-images.githubusercontent.com/59161083/216928817-dffec866-bafe-49f4-8c89-9e82621d7807.png"/></a>
            <br/>
        </td>
    </tr>
</table>

위의 사진들은 '백종원의 쿠킹로그'라는 유튜브 채널에서 가져온 것으로, 일반인의 표정과 대응되는 백종원의 사진으로 대체함으로써 그들의 반응을 효과적으로 파악할 수 있도록 하였다. 
하지만, 이러한 방식은 편집자가 직접 해당 프레임의 얼굴을 찾아 바꿔주어야 하기 때문에 상당한 비용(시간, 노력 등)이 발생한다.
이에 우리는 기존의 모자이크를 대체하여 사람을 특정할 수 있을 정도로 얼굴을 노출시키지 않는 동시에 얼굴 표정, 시선, 눈빛과 같은 정보는 보존할 수 있는 새로운 방식, **CAFE(CArtoonize For Extra faces)**를 제안한다.

## 프로젝트 환경
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Architecture
<table>
    <th colspan=1></th>
        <td align="center">
            <a href="https://github.com/SeongSuKim95"><img height="500px" width="1000px" src="https://user-images.githubusercontent.com/59161083/216927654-a4676796-80a2-4802-bdba-97449debbf39.png"/></a>
            <br/>
        </td>
    </tr>
</table>

## Usage

## Wrap-Up Report

## Reference