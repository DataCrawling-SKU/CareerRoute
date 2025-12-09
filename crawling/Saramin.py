from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import csv
import time

backendResult = [] # 데이터 행 저장

# 셀레니움 옵션
options = Options()
options.add_argument("window-size=1920x1080")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 첫 페이지 열기
base_url = "https://www.saramin.co.kr/zf_user/search/recruit?search_area=main&search_done=y&search_optional_item=n&searchType=search&searchword=백엔드"
driver.get(base_url)
time.sleep(2)

page = 1 # 1페이지부터 크롤링

while True: # 페이지 당 100개씩 29 페이지까지 있음
  print(f"{page} 페이지 접속 중")

  #url = f"https://www.saramin.co.kr/zf_user/search/recruit?ajax=y&searchType=search&searchword=백엔드&recruitPage={page}&recruitSort=relation"

  soup = BeautifulSoup(driver.page_source, "html.parser")
  print(f"{page} 페이지 크롤링 중")

  items = soup.select("div.item_recruit")

  if not items: # item_recruit 태그의 데이터가 더이상 없어서 리스트가 빈 경우 종료
    print("크롤링할 데이터가 없습니다. 크롤링을 종료합니다.")
    break

  for item in items: # 채용공고 리스트를 돌면서 필요한 태그의 텍스트 추출
    # 회사 이름
    company_tag = item.select_one(".corp_name a") # a 태그 추출
    company = company_tag.text.strip() if company_tag else "" # 3항 연산자로 태그 내부의 텍스트를 꺼내고 공백 제거

    # 카테고리
    category_tag = item.select(".job_sector a") # 리스트
    categories = [c.text.strip() for c in category_tag] # 위의 리스트를 돌면서 공백 제거 후 다시 리스트로 저장

    # 날짜
    date_tag = item.select_one(".job_sector .job_day") # 채용공고 등록일 or 수정일
    # 날짜 앞에 "등록일" 혹은 "수정일" 모두 제거 후 날짜만 저장
    if date_tag:
      date_text = date_tag.text.strip() # 공백 제거
      date = date_text.replace("등록일", "").replace("수정일","").strip() # 등록일 수정일 문자열 제거
    else :
      date = ""

    # 시계열 형태로 csv에 카테고리 당 한 행으로 저장
    for c in categories:
      backendResult.append({
        "company" : company,
        "category" : c,
        "date" : date   
      })
  # 다음 페이지로 이동
  try:
      # 현재 페이지 span을 읽고 다음 페이지 계산
      current_page_span = soup.select_one("span.page")
      current_page_num = int(current_page_span.text.strip())
      next_page_num = current_page_num + 1

      # 다음 번호 a 태그 클릭
      next_a = driver.find_elements(By.CSS_SELECTOR, f'a.page[page="{next_page_num}"]')
      if next_a:
        next_a[0].click()
      else:
        # 다음 페이지의 a 태그가 없다면 '다음'버튼인 btnNext 클릭 (10배수 페이지 넘어갈때)
        btn_next = driver.find_elements(By.CSS_SELECTOR, "a.btnNext")
        if btn_next:
          btn_next[0].click()
        else:
          print("다음 페이지 없음 -> 크롤링 종료")
          break
      time.sleep(2)
      page += 1

  except Exception as e:
      print("페이지 이동 중 오류 : ", e)
      break

# CSV 저장
csv_file = '/Users/parkjuyong/Desktop/4-1/CareerRoute/assets/saramin_backend.csv'

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["company", "category", "date"])
    writer.writeheader()
    writer.writerows(backendResult)

print(f"\n총 {len(backendResult)}개의 레코드 저장됨.")
driver.quit()