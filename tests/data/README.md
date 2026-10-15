# Antoni 기준값

`antoni_reference.npz`는 Python 구현의 출력으로 만든 기준값이 아니다.
Antoni의 MATLAB 계산을 별도로 실행한 결과와 해당 입력을 보관한다.

## 출처

- 원본: [Fast Kurtogram, Jérôme Antoni](https://www.mathworks.com/matlabcentral/fileexchange/48912-fast-kurtogram)
- 패키지 버전: 1.0.0.0, 2015-05-18 배포
- 소스의 Last Revision: 12-2014
- `Fast_Kurtogram.m` SHA-256: `d24ec1b9903063918ac6acece2f54d6ee92041724f786308bdba3957e451aac9`
- 실행 환경: MATLAB `26.2.0.3386108 (R2026b)`
- 사용 조건: [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)

원본에서 그래프·대화형 입력만 제외하고 `Kwav`, `h`, `g`, `h1`, `h2`, `h3`를
출력하도록 진입 함수의 반환값을 바꿨다.
SK, 이진·삼진 필터링, 재귀 트리 계산 함수는 원본을 그대로 실행했다.

Signal Processing Toolbox가 없어 MATLAB 내장 `fir1`을 실행하지 못했다.
기준 실행의 `fir1(order, cutoff)`는 다음 공식을 MATLAB에서 독립 계산했다.
SciPy에서 만든 계수를 MATLAB에 넘기지 않았다.

1. `m = (0:order) - order/2`
2. 이상 저역통과 응답: `sin(pi * cutoff * m) / (pi * m)`; `m = 0`에서는 `cutoff`
3. 대칭 Hamming 창: `0.54 - 0.46 * cos(2*pi * (0:order) / order)`
4. 두 응답을 곱하고 계수 합으로 나눠 DC 이득을 1로 정규화

이는 이번 원본의 스칼라 저역통과 설계에 해당한다.
MATLAB 내장 `fir1`의 직접 실행까지 검증한 기준값은 아니다.

## 포함한 검사 입력

| 이름 | 샘플 수 | 이진 최대 레벨 | 목적 |
|---|---:|---:|---|
| white_noise | 8192 | 5 | 잡음의 SK와 음수 클리핑 |
| impacts | 8192 | 5 | 반복 충격과 최적 대역 선택 |
| odd_length | 5001 | 5 | 다운샘플링의 시작 위치와 홀수 길이 처리 |
| level_one | 8192 | 1 | 이진 행만 있는 경우 |
| level_two | 8192 | 2 | 첫 삼진 행의 삽입 위치 |

모든 입력의 샘플링 주파수는 12000 Hz다.
합성 입력은 NumPy `default_rng(2007)`의 표준정규 잡음으로 생성했다.
반복 충격은 113샘플 간격, 공진 3250 Hz, 감쇠 계수 700의 정현파에
진폭 0.15의 잡음을 더했다. 정확한 입력 배열을 함께 저장하므로
회귀 검사는 난수 생성기 버전이나 재생성에 의존하지 않는다.

원본 MATLAB은 각 대역의 SK를 여러 표시 열에 반복해 저장한다.
검사에서는 Python의 대역별 값을 같은 형식으로 확장해 전체 행을 비교한다.
허용 오차는 SK의 상대·절대 오차 각각 `1e-11`이다.

실제 CWRU 입력 40개도 동일한 MATLAB 계산과 대조했다.
45개 입력 전체에서 최대 SK 차이는 `2.5011104298755527e-12`,
최대 필터 계수 차이는 `1.0303264460093434e-15`였다.
현재 로컬 실행의 상세 결과는 `results/antoni_validation/validation.json`에 저장한다.

실행:

```powershell
python -B -m unittest discover -s tests -v
```
