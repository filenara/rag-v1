import pdfplumber
import json
import re

PDF_PATH = "blackwell-datasheet-ultra-blackwell-4169750.pdf"
START_PAGE = 103
END_PAGE = 382

# Gürültü ve İstenmeyen Desenler
IGNORE_KEYWORDS = [
    "issue 7", "2017-01-25", "(part of speech)", "word", "blank page", 
    "approved meaning/", "approved example", "not approved", "page 2-1"
]

def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()

def extract_alternatives(text):
    """Metin içindeki BÜYÜK HARFLİ alternatifleri bulur."""
    if not text: return []
    pattern = r'([A-Z]{2,}(?:\s+[A-Z]+)*)\s*(?:\([a-z]+\))?'
    matches = re.finditer(pattern, text)
    alts = []
    for m in matches:
        full_match = m.group(0).strip()
        if (len(full_match) > 1 or full_match in ["A", "I"]) and "NOTE" not in full_match:
            if full_match not in alts:
                alts.append(full_match)
    return alts

def is_real_keyword(text):
    text = text.strip()
    if not text: return False
    # Keyword içinde ':' olmaz 
    if ":" in text: return False
    # Keyword genelde kısa olur 
    if len(text.split()) > 4: return False
    # "..." ile bitiyorsa veya başlıyorsa devam metnidir
    if text.endswith("...") or text.startswith("..."): return False
    return True

def parse_ste100_v12(pdf_path, start_page, end_page):
    results = []
    
    current_entry = {
        "keyword": None,
        "pos": None,
        "is_approved": False,
        "desc_buffer": [],
        "ste_ex_buffer": [],
        "non_ste_ex_buffer": []
    }

    def flush_entry():
        if not current_entry["keyword"]: return

        full_desc = clean_text(" ".join(current_entry["desc_buffer"]))
        full_ste = clean_text(" ".join(current_entry["ste_ex_buffer"]))
        full_non_ste = clean_text(" ".join(current_entry["non_ste_ex_buffer"]))

        alternatives = []
        if not current_entry["is_approved"]:
            alternatives = extract_alternatives(full_desc)

        examples = []
        if full_ste or full_non_ste:
            examples.append({
                "ste": full_ste,
                "non_ste": full_non_ste
            })

        results.append({
            "keyword": current_entry["keyword"],
            "part_of_speech": current_entry["pos"],
            "is_approved": current_entry["is_approved"],
            "approved_alternatives": alternatives,
            "rule_description": full_desc,
            "examples": examples
        })

    print(f"İşlem başlıyor: {pdf_path}...")

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page - 1, end_page): 
            page = pdf.pages[i]
            width = page.width
            height = page.height
            
            # --- 1. KIRPMA (CROP) ---
            # Footer'ı biraz daha az kesiyoruz (60 -> 50)
            # Genişliği 'width' olarak tam alıyoruz.
            # Üstten 80 piksel kesmek header için yeterli.
            cropped_page = page.crop((0, 80, width, height - 50))
            
            # --- 2. TABLO ÇIKARMA ---
            # 'edge_min_length': Kenardaki çok küçük çizgileri yoksayar.
            # 'snap_tolerance': Kelimelerin hizalanma toleransı (3 -> 4 yaptık).
            table = cropped_page.extract_table({
                "vertical_strategy": "text", 
                "horizontal_strategy": "text",
                "snap_tolerance": 4, 
                "intersection_tolerance": 3
            })

            if not table: continue

            for row in table:
                clean_row = [cell.strip().replace('\n', ' ') if cell else "" for cell in row]
                
                # Boş satırları atla
                if not any(clean_row): continue
                
                # Sütunları güvenli al (Eksik sütun varsa doldur)
                col_word = clean_row[0] if len(clean_row) > 0 else ""
                col_meaning = clean_row[1] if len(clean_row) > 1 else ""
                col_ste = clean_row[2] if len(clean_row) > 2 else ""
                # Eğer 4. sütun yoksa (pdfplumber bazen sağ tarafı almaz), boş string ata
                col_non = clean_row[3] if len(clean_row) > 3 else ""

                # Gürültü Filtresi
                is_noise = False
                for pattern in IGNORE_KEYWORDS:
                    if pattern in col_word.lower():
                        is_noise = True; break
                if is_noise: continue

                # --- KARAR MEKANİZMASI ---
                is_new_entry = False
                
                if col_word:
                    # 1. Sütun dolu ama bu gerçekten keyword mü?
                    if is_real_keyword(col_word):
                        is_new_entry = True
                    else:
                        # Değilse description buffer'a ekle.
                        if current_entry["keyword"]:
                            current_entry["desc_buffer"].append(col_word)
                
                if is_new_entry:
                    flush_entry() # Eskiyi kaydet
                    
                    # Kelime Analizi
                    match = re.match(r"([^\(]+)\s*(\(([a-z]+)\))?", col_word)
                    if match:
                        raw_keyword = match.group(1).strip()
                        pos = match.group(3) if match.group(3) else "unknown"
                    else:
                        raw_keyword = col_word.strip()
                        pos = "unknown"

                    keyword_lower = raw_keyword.lower().replace(",", "")
                    is_approved = raw_keyword.isupper()

                    current_entry = {
                        "keyword": keyword_lower,
                        "pos": pos,
                        "is_approved": is_approved,
                        "desc_buffer": [col_meaning] if col_meaning else [],
                        "ste_ex_buffer": [col_ste] if col_ste else [],
                        "non_ste_ex_buffer": [col_non] if col_non else []
                    }
                
                else:
                    # Devam satırı
                    if current_entry["keyword"]:
                        if col_meaning: current_entry["desc_buffer"].append(col_meaning)
                        if col_ste: current_entry["ste_ex_buffer"].append(col_ste)
                        if col_non: current_entry["non_ste_ex_buffer"].append(col_non)

    flush_entry()
    return results

# --- ÇALIŞTIRMA ---
try:
    data = parse_ste100_v12(PDF_PATH, START_PAGE, END_PAGE)
    output_file = "ste100_rules_v12.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nBAŞARILI! '{output_file}' oluşturuldu.")
    print(f"Toplam kural sayısı: {len(data)}")
    
    # Kontrol
    print("\n--- ÖRNEK ÇIKTI (İLK 3) ---")
    print(json.dumps(data[:3], indent=2, ensure_ascii=False))

except Exception as e:
    print(f"Hata oluştu: {e}")