class TurkishTokenizer:
    """
    Türkçe ASR (Otomatik Konuşma Tanıma) modelleri için karakter tabanlı tokenizer sınıfı.

    Bu sınıf, Türkçe metinleri modelin anlayabileceği sayısal ID'lere (kodlama) ve 
    model çıktısı olan ID dizilerini tekrar okunabilir metne (kod çözme) dönüştürür.
    CTC (Connectionist Temporal Classification) kaybı ile uyumlu çalışacak şekilde tasarlanmıştır.

    Özellikler:
    - ID 0, her zaman CTC için gerekli olan 'blank' (boş) token'a ayrılmıştır ("-").
    - Boşluk karakteri " " dahil olmak üzere Türkçe alfabesindeki harfleri destekler.
    """

    def __init__(self):
        """
        Tokenizer sınıfını başlatır ve karakter sözlüğünü oluşturur.
        
        Sözlük Yapısı:
        - "-" (Blank): 0
        - " " (Boşluk): 1
        - a, b, c, ... : 2, 3, 4 ...
        """
        # "-" : CTC blank token
        # " " : kelimeler arası boşluk
        # Geri kalanı Türkçe alfabesi
        self.chars = ["-", " "] + list("abcçdefgğhıijklmnoöprsştuüvyz")

        # Karakterden ID'ye dönüşüm sözlüğü (Encode için)
        self.char_to_id = {
            char: i for i, char in enumerate(self.chars)
        }

        # ID'den karaktere dönüşüm sözlüğü (Decode için)
        self.id_to_char = {
            i: char for i, char in enumerate(self.chars)
        }

    def encode(self, text):
        """
        Verilen metni karakter ID dizisine (vektörüne) dönüştürür.

        İşlemler:
        1. Metni küçük harfe çevirir.
        2. Sözlükte tanımlı olmayan karakterleri (noktalama işaretleri vb.) atlar.
        
        Args:
            text (str): Dönüştürülecek ham metin.

        Returns:
            List[int]: Metnin sayısal temsili olan ID listesi.
        """
        return [
            self.char_to_id[c]
            for c in text.lower()
            if c in self.char_to_id
        ]

    def decode(self, ids):
        """
        Sayısal ID dizisini doğrudan metne dönüştürür (Raw Decode).

        Not: Bu yöntem CTC mantığını (tekrarları silme vb.) uygulamaz.
        Genellikle modelin ham çıkışını hata ayıklamak (debug) için kullanılır.

        Args:
            ids (Iterable[int]): Sayısal ID dizisi.

        Returns:
            str: Dönüştürülmüş metin.
        """
        return "".join(
            self.id_to_char[i]
            for i in ids
            if i in self.id_to_char
        )

    def ctc_decode(self, ids):
        """
        CTC (Connectionist Temporal Classification) çıkışını anlamlı metne dönüştürür
        (Greedy Search yöntemi ile).

        CTC Kod Çözme Mantığı:
        1. Ardışık tekrar eden karakterleri tek'e indirger (Örn: "aa" -> "a").
        2. 'Blank' tokenları (ID: 0) temizler.

        Örnek:
           Model Çıkışı: [A, A, -, -, B, B, B, -, A]
           Adım 1 (Collapse): [A, -, B, -, A]
           Adım 2 (Remove Blank): [A, B, A] -> "aba"

        Args:
            ids (Iterable[int]): Modelin çıkışından (argmax sonrası) elde edilen ID dizisi.

        Returns:
            str: Nihai anlamlı metin.
        """
        decoded = []
        last_id = None

        for curr_id in ids:
            # Aynı ID ardışık gelmişse atla (Collapse)
            if curr_id != last_id:
                # Blank token değilse karakteri ekle (ID 0 = Blank)
                if curr_id != 0: 
                    decoded.append(self.id_to_char[curr_id])
            last_id = curr_id

        return "".join(decoded)
