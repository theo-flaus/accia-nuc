import fitz
import re
import numpy as np

class PyMuExtract:

    def __init__(self, path: bytes) -> None:
        #self.doc = fitz.open(path)
        self.doc = fitz.open(stream=path, filetype='pdf')
        self.paragraphs = self._extract_paragraphs(self.doc)
        self.paragraphs = self._clean_paras(self.paragraphs)
        self.paragraphs = self._regroup_paras(self.paragraphs)
        

    @staticmethod
    def _extract_paragraphs(doc: fitz.Document) -> list[str]:
        paras = []
        for page in doc:
            paras_page = [p[4] for p in page.get_text('blocks')]
            paras.extend(paras_page)
        return paras

    @staticmethod
    def _clean_paras(paragraphs: list[str]) -> list[str]:
        paragraphs_cleaned = []
        patterns = [
            r"Created with an evaluation copy of[\s\S]*words/",
            r"^<image:.*>$",
            r"^Source.*",
            r"^Rev[\s\S]*Approuvé par.*$",
            r"^\d{2}\s+\d{2}\/\d{2}\/\d{2}[\s\S].*",
            r"^Evaluation Only*",
            r"Ecole du nucléaire[\s\S]*",
            r".*Created with Aspose.*",
            r"IMAGE",
            r"Insérer un bandeau",
            r".*accident de.* est classé au plus haut niveau.*",
            r".*accident de.* a été classé au niveau.*",
            r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
            r"^[^\w]*$"

        ]
        for p in paragraphs:
            for pattern in patterns:
                p = re.sub(pattern, '', p)
            #if len(p.split(' ')) > 1:
            paragraphs_cleaned.append(p.strip())
            
        

        return list(filter(None,paragraphs_cleaned))

    @staticmethod
    def _regroup_paras(paragraphs: list[str]) -> list[str]:
        indexes_short_paras = []
        for i,p in enumerate(paragraphs):
            if len(p.split(' ')) <= 7:
                indexes_short_paras.append([i, i+1])

        indexes_short_paras = np.array(indexes_short_paras)
        
        result = []
        for i, string in enumerate(paragraphs):
            found = False
            for index in indexes_short_paras:
                if index[0] <= i <= index[1]:
                    found = True
                    if i == index[0]:
                        result.append(string)
                    else:
                        result[-1] += f'\n\n{string}'
                    break
            if not found:
                result.append(string)

        return result
