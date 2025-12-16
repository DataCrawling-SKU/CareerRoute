from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
import pandas as pd

# 셀레니움 옵션
options = Options()
options.add_argument("--start-maximized")
options.add_argument("window-size=1920x1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# tiobe의 해당 언어의 차트 페이지
print("웹 브라우저 접속 중")
driver.get("https://www.tiobe.com/tiobe-index/r/")
print("웹 브라우저 접속 성공")

WebDriverWait(driver, 10).until( # 페이지는 렌더링 되어도 차트는 비동기로 렌더링 되기 때문에 차트가 렌더링 되기까지 10초 대기
  lambda d: d.execute_script( # 브라우저에서 js코드 실행
    # Hightchart 라이브러리가 로드 되었는지, 차트가 최소 1개 이상 렌더링 되었는지 확인
    "return typeof Highcharts !== 'undefined' && Highcharts.charts.length > 0" # js코드를 실행 시키는 부분
  )
)
print("차트 렌더링 성공")

print("차트 데이터 크롤링 중")
data = driver.execute_script(
  """
  return Highcharts.charts[0].series[0].data.map(p => ({ 
      timestamp: p.x,
      percent: p.y
  }));
  """
)
print("크롤링 완료")

driver.quit() # 브라우저 종료

# csv로 저장
print("csv파일로 저장중")
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["timestamp"], unit="ms") # 수집한 timestamp의 값을 dataTime 객체로 변환(초단위까지 필요 없음)
df = df[["date","percent"]]

df.to_csv('/Users/parkjuyong/Desktop/4-1/CareerRoute/assets/tiobe/r.csv', index=False)

print("완료")