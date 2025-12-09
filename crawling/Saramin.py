from bs4 import BeautifulSoup
import requests
import csv

backendResult = [] # 데이터 행 저장

page = 1 # 1페이지부터 크롤링

# 외부 페이지 크롤링 시 봇 차단 문제 해결
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

while True: # 페이지 당 100개씩 29 페이지까지 있음
  # URL 문자열을 문자열 포멧대신 request param으로 관리(가독성, 버그 낮춤)
  params = {
        "search_area": "main",
        "search_done": "y",
        "search_optional_item": "n",
        "searchType": "search",
        "searchword": "백엔드",
        "recruitPage": page,
        "recruitSort": "relation",
        "recruitPageCount": 100
    }

  base = "https://www.saramin.co.kr/zf_user/search/recruit"
  response = requests.get(base, params=params)
  soup = BeautifulSoup(response.text, "html.parser")

  print(response.text[:3000])
  print(f"{page} 페이지 크롤링 중")

  items = soup.select("div.item_recruit")
  if not items: # item_recruit 태그의 데이터가 더이상 없어서 리스트가 빈 경우 종료
    print("크롤링할 데이터가 없습니다. 크롤링을 종료합니다.")
    break

  for item in items: # 채용공고 리스트를 돌면서 필요한 태그의 텍스트 추출
    # 회사 이름
    company_tag = item.select(".corp_name a") # a 태그 추출
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
    
    page += 1 # 다음 페이지

# CSV 저장
csv_file = '/Users/parkjuyong/Desktop/4-1/CareerRoute/assets/saramin_backend.csv'

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["company", "category", "date"])
    writer.writeheader()
    writer.writerows(backendResult)

print(f"\n총 {len(backendResult)}개의 레코드 저장됨.")