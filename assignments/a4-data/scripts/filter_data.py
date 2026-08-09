from fastwarc.warc import ArchiveIterator , WarcRecordType
import gzip
from cs336_data.html2text import extract_text_from_html_bytes
from cs336_data.languageIdentification import identify_language
from cs336_data.Pii import mask_emails,mask_phone_numbers,mask_ips
from cs336_data.harmful import identify_nsfw, identify_hate
from cs336_data.quality import gopher_quality_filter
from cs336_data.quality_classifier import classify_quality
from cs336_data.exactlinededup import exact_line_dedup
from cs336_data.minHash_LSH_dedup import dedup

def process_single_wet_file(input_path: str, output_path: str):
    stats = {
        "total": 0,
        "gopher_rejected": 0,
        "kept": 0,
        "emails_masked": 0,
        "phones_masked": 0,
        "ips_masked": 0,
        "nsfw_rejected": 0,
        "toxic_rejected": 0,
        "language_identified": 0,
    }
    
    with gzip.open(input_path, 'rb') as f_in, open(output_path, 'w', encoding = "utf-8") as f_out:
        
        for record in ArchiveIterator(f_in):
            if record.record_type != WarcRecordType.conversion:
                continue
            
            stats["total"] += 1
            
            text = record.reader.read().decode('utf-8', errors='replace')
            
            # Apply gopher quality filter
            if not gopher_quality_filter(text):
                stats["gopher_rejected"] += 1
                continue
            
            # Identify language
            language, confidence = identify_language(text)
            if language != "en" and confidence > 0.7:
                stats["language_identified"] += 1
                continue

            # NSFW
            label, score = identify_nsfw(text)
            if label == "nsfw" and score > 0.9:
                stats["nsfw_rejected"] += 1
                continue
            
            # toxic
            label, score = identify_hate(text)
            if label == "toxic" and score > 0.9:
                stats["toxic_rejected"] += 1
                continue            
            

            # Mask PII
            text, emails_masked = mask_emails(text)
            stats["emails_masked"] += emails_masked

            text, phones_masked = mask_phone_numbers(text)
            stats["phones_masked"] += phones_masked

            text, ips_masked = mask_ips(text)
            stats["ips_masked"] += ips_masked
            
            text = " ".join(text.split())
            
            f_out.write(text + "\n")
            stats["kept"] += 1
            
    return stats

if __name__ == "__main__":
    stats = process_single_wet_file(
        "local-shared-data/CC/example.warc.wet.gz",
        "data/filtered/sample.txt"
    )
    print("Processing complete. Stats:", stats)