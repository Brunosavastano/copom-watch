from copom_tone_index.text import clean_copom_html, split_sentences


def test_clean_copom_html_removes_institutional_blocks() -> None:
    raw = """
    <div>
      <p>O ambiente externo permanece incerto e exige cautela.</p>
      <p>Votaram por essa decisão os seguintes membros do Comitê.</p>
      <div id="ata_info"><p>Presentes: Diretor A</p></div>
      <table><tr><td>IPCA</td></tr></table>
    </div>
    """
    clean = clean_copom_html(raw)
    assert "ambiente externo" in clean
    assert "Votaram" not in clean
    assert "Presentes" not in clean
    assert "IPCA" not in clean


def test_split_sentences_preserves_common_abbreviations() -> None:
    text = "O Copom reduziu a Selic em 0,25 p.p. A taxa ficou em 10,50% a.a. O comitê segue cauteloso."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert "p.p." in sentences[0]
    assert "a.a." in sentences[1]
