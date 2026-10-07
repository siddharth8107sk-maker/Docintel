import pypdf

reader = pypdf.PdfReader("data/docs/Practicum 4.pdf")
print("Total pages:", len(reader.pages))
for i in range(min(5, len(reader.pages))):
    text = reader.pages[i].extract_text() or ""
    print(f"\n--- PAGE {i+1} ---")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:10]:
        print(line)
