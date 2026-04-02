import requests
import time

BASE_URL = "http://127.0.0.1:8765/api"

def test_pagination():
    print("Testing /api/books pagination...")
    r = requests.get(f"{BASE_URL}/books?page=1&limit=5")
    data = r.json()
    print(f"Page 1: {len(data['books'])} books, Total in system: {data['total']}")
    assert len(data['books']) <= 5
    
def test_semantic_search():
    print("\nTesting /api/search semantic relevance...")
    # "Network Security" should ideally return "Cryptography" if semantic is working
    r = requests.get(f"{BASE_URL}/search?q=network security&limit=10")
    data = r.json()
    print(f"Search 'network security' found {data['total']} results (Semantic: {data.get('semantic')})")
    titles = [b['title'].lower() for b in data['books']]
    for t in titles[:5]:
        print(f" - {t}")
    
    # "Textbook" Breadth test
    r2 = requests.get(f"{BASE_URL}/search?q=textbook&limit=10")
    data2 = r2.json()
    print(f"\nSearch 'textbook' found {data2['total']} results")
    for t in [b['title'] for b in data2['books'][:5]]:
        print(f" - {t}")

if __name__ == "__main__":
    try:
        test_pagination()
        test_semantic_search()
        print("\nBackend Verification PASSED")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
