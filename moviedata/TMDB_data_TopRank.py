import requests
import time
import json

# TMDb API 키
api_key = ''  # TMDb API 키를 입력하세요

# 가져오려는 영화의 개수
total_movies = 1000  # 가져올 영화의 수 (예: 100개 영화)
top_movies = []
page = 1  # 첫 페이지부터 시작
movies_per_page = 20  # 한 페이지에 가져오는 영화 수 (최대 20개)

# 전체 페이지 수 계산
total_pages = (total_movies // movies_per_page) + (1 if total_movies % movies_per_page != 0 else 0)

# 장르 목록 가져오기 (장르 ID와 이름 매핑)
genre_url = f'https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US'
genre_response = requests.get(genre_url)
genre_data = genre_response.json()

# 장르 ID와 이름을 매핑하는 딕셔너리
genre_dict = {genre['id']: genre['name'] for genre in genre_data['genres']}

while page <= total_pages:
    url = f'https://api.themoviedb.org/3/discover/movie?api_key={api_key}&language=en-US&sort_by=popularity.desc&page={page}'

    response = requests.get(url)
    data = response.json()

    for movie in data['results']:
        movie_id = movie['id']
        title = movie['title']
        overview = movie['overview']

        # 장르 ID를 장르 이름으로 변환
        genres = [genre_dict.get(genre_id, 'Unknown') for genre_id in movie.get('genre_ids', [])]

        release_date = movie['release_date']
        vote_average = movie['vote_average']

        # 영화 키워드 정보 가져오기
        keywords_url = f'https://api.themoviedb.org/3/movie/{movie_id}/keywords?api_key={api_key}'
        keywords_response = requests.get(keywords_url)
        keywords_data = keywords_response.json()
        keywords = [keyword['name'] for keyword in keywords_data['keywords']]  # 키워드 정보

        # 영화 정보 저장
        movie_info = {
            'title': title,
            'overview': overview,
            'genres': genres,  # 장르 이름으로 저장
            'release_date': release_date,
            'vote_average': vote_average,
            'keywords': keywords  # 키워드 정보 추가
        }
        top_movies.append(movie_info)

        # 원하는 영화 개수가 충분히 채워졌으면 종료
        if len(top_movies) >= total_movies:
            break

    # 페이지를 바꾸기 전에 2초 쉬기
    time.sleep(2)

    # 데이터를 JSON 파일로 저장
    with open(f'./top{total_movies}_movies.json', 'w', encoding='utf-8') as f:
        json.dump(top_movies, f, ensure_ascii=False, indent=4)

    print(f"Top {len(top_movies)} movies data has been saved to 'top{total_movies}_movies.json'")

    page += 1

# 가져온 데이터 출력
for movie in top_movies:
    print(f"Title: {movie['title']}")
    print(f"Overview: {movie['overview']}")
    print(f"Genres: {', '.join(movie['genres'])}")  # 장르 이름 출력
    print(f"Release Date: {movie['release_date']}")
    print(f"Average Vote: {movie['vote_average']}")
    print(f"Keywords: {', '.join(movie['keywords'])}")  # 키워드 출력
    print('-' * 50)
