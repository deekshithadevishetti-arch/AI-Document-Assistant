def create_chunks(text, chunk_size=2):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chunks = []

    for i in range(0, len(lines), chunk_size):
        chunk = " ".join(lines[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def chunk_text(text, chunk_size=2):

    return create_chunks(text, chunk_size)