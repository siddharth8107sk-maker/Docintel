import pypdf

reader = pypdf.PdfReader("data/docs/Software engg URK25CS1154 TEST-1.pdf")
img_data = reader.pages[0].images[0].data
with open("scratch/page1.jpg", "wb") as f:
    f.write(img_data)
print("Saved scratch/page1.jpg, size:", len(img_data))
