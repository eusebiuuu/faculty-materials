import hashlib
import requests
import os
from dotenv import load_dotenv

# Încărcăm variabilele din fișierul .env
load_dotenv()

def get_sha256(file_path):
    """Calculează hash-ul SHA256 al unui fișier."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Eroare: Fișierul '{file_path}' nu a fost găsit.")
        return None

def check_virustotal(file_hash, api_key):
    """Interoghează VirusTotal API v3 folosind hash-ul."""
    if not api_key:
        print("Eroare: Cheia API nu a fost găsită în fișierul .env")
        return

    url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {
        "accept": "application/json",
        "x-apikey": api_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        stats = data['data']['attributes']['last_analysis_stats']
        
        print("-" * 30)
        print(f"Rezultate VirusTotal pentru: {file_hash}")
        print(f"Malicious:   {stats['malicious']}")
        print(f"Suspicious:  {stats['suspicious']}")
        print(f"Undetected:  {stats['harmless']}")
        print(f"Total scanări: {sum(stats.values())}")
        print("-" * 30)
    elif response.status_code == 404:
        print("\n[!] Hash-ul nu există în baza de date VirusTotal (fișierul nu a mai fost scanat).")
    elif response.status_code == 401:
        print("\n[!] Eroare: Cheie API invalidă.")
    else:
        print(f"\nEroare API: {response.status_code}")

if __name__ == "__main__":
    # Citim cheia din variabila de mediu încărcată anterior
    api_key = os.getenv("VT_API_KEY")
    
    # Poți schimba calea fișierului aici
    nume_fisier = "malware.png" 
    
    hash_rezultat = get_sha256(nume_fisier)
    
    if hash_rezultat:
        print(f"SHA256 calculat: {hash_rezultat}")
        check_virustotal(hash_rezultat, api_key)