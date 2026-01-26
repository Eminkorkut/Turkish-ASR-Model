import numpy as np

def pre_emphasis(signal, alpha=0.97):
    """
    Ses sinyaline Pre-emphasis filtresi uygular.

    Amaç:
    - Konuşma sinyalinde doğal olarak zayıf olan yüksek frekanslı bileşenleri güçlendirir.
    - Sinyalin spekral dengesini iyileştirerek daha iyi özellik çıkarımı sağlar.
    
    Formül: y[n] = x[n] - alpha * x[n-1]

    Args:
        signal (np.ndarray): Giriş ses sinyali (1D dizi).
        alpha (float): Filtre katsayısı (genelde 0.95 - 0.97 arası).

    Returns:
        np.ndarray: Filtrelenmiş sinyal.
    """
    # Sinyalin kendisi ile bir adım kaydırılmış halinin farkını alır
    return np.append(signal[0], signal[1: ] - alpha * signal[:-1])

def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    """
    Ses sinyalini kısa, örtüşen (overlapping) çerçevelere (frames) böler.

    Neden?: Konuşma sinyali durağan değildir (non-stationary), ancak çok kısa sürelerde (örn. 25ms)
    durağan kabul edilebilir. FFT (Hızlı Fourier Dönüşümü) bu kısa parçalara uygulanır.

    Args:
        signal (np.ndarray): Giriş ses sinyali.
        sample_rate (int): Örnekleme hızı (Hz).
        frame_size (float): Bir çerçevenin süresi (saniye, örn: 0.025 = 25ms).
        frame_stride (float): Çerçeveler arası kayma adımı (saniye, örn: 0.01 = 10ms).

    Returns:
        np.ndarray: (Çerçeve Sayısı, Çerçeve Uzunluğu) boyutunda matris.
    """
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)

    # Toplam çerçeve sayısını hesapla
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

    # Sinyalin sonunu doldur (Padding)
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    # İndeks matrisini oluştur (Vektörel işlem)
    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1)) +
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )

    # Çerçeveleri çek
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def windowing(frames):
    """
    Her bir çerçeveye Hamming penceresi (Windowing) uygular.

    Amaç:
    - Çerçevelerin başındaki ve sonundaki keskin kesinti etkisini yumuşatmak.
    - Spektral sızıntıyı (spectral leakage) azaltmak.

    Args:
        frames (np.ndarray): Çerçevelenmiş sinyal matrisi.

    Returns:
        np.ndarray: Pencerelenmiş çerçeveler.
    """
    frame_length = frames.shape[1]
    # Hamming penceresi fonksiyonu
    window = 0.54 - (0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1)))
    return frames * window

def power_spectrum(frames, nfft=512):
    """
    Çerçevelerin güç spektrumunu (Power Spectrum) hesaplar.
    
    Adımlar:
    1. FFT (Hızlı Fourier Dönüşümü) uygulanır.
    2. Mutlak değer (genlik) alınır.
    3. Karesi alınarak güce dönüştürülür.

    Args:
        frames (np.ndarray): Pencerelenmiş çerçeveler.
        nfft (int): FFT nokta sayısı (genellikle 512, 1024, 2048).

    Returns:
        np.ndarray: Güç spektrumu matrisi.
    """
    # Real FFT -> Genlik
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    # Genlik -> Güç
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)
    return pow_frames

def get_filter_banks(pow_frames, sample_rate, nfilt=40, nfft=512):
    """
    Güç spektrumundan Mel Filterbank (Log-Mel Spektrogram) özelliklerini çıkarır.

    Bu özellikler, insan kulağının frekans algısını (Mel ölçeği) taklit eder ve
    ASR modelleri için en yaygın kullanılan girdidir.

    Args:
        pow_frames (np.ndarray): Güç spektrumu.
        sample_rate (int): Örnekleme hızı (Hz).
        nfilt (int): Mel filtre sayısı (Öznitelik vektör boyutu).
        nfft (int): FFT nokta sayısı.

    Returns:
        np.ndarray: Log-Mel Filterbank matrisi (Zaman, nfilt).
    """
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    
    # Mel ekseninde eşit aralıklı noktalar
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    # Hz'e çevir
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    # FFT bin indekslerine çevir
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    
    # Üçgen filtreleri oluştur
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # Sol sınır
        f_m = int(bin[m])             # Tepe
        f_m_plus = int(bin[m + 1])    # Sağ sınır

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    # Filtre bankasını uygula
    filter_banks = np.dot(pow_frames, fbank.T)
    # Logaritma hatasını önlemek için epsilon ekle
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # dB (Log) ölçeğine geç
    filter_banks = 20 * np.log10(filter_banks)
    
    return filter_banks

def normalize_features(features):
    """
    Özellikleri normalize eder (Cepstral Mean and Variance Normalization - CMVN).

    Her frekans bandı için ortalamayı 0, varyansı 1 yapar.
    Bu, modelin eğitimi için kritik öneme sahiptir.

    Args:
        features (np.ndarray): Özellik matrisi.

    Returns:
        np.ndarray: Normalize edilmiş özellikler.
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-8)
