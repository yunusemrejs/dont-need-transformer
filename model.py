
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- Model Classes ---

class LIFActivation(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Nöron Aktivasyon Fonksiyonu
    
    Bu sınıf, biyolojik nöronların davranışını taklit eden LIF modelini uygular.
    Matematiksel olarak:
    V(t) = leak * V(t-1) + I(t)  # Membran potansiyeli güncelleme
    Spike = 1 if V(t) >= threshold else 0  # Aksiyon potansiyeli üretimi
    V(t) = 0 if Spike == 1 else V(t)  # Reset mekanizması
    
    Parametreler:
    - threshold (float): Aksiyon potansiyeli üretimi için eşik değeri
    - leak (float): Sızıntı faktörü, önceki potansiyelin ne kadarının korunacağını belirler
    """
    def __init__(self, threshold=1.0, leak=0.95):
        super(LIFActivation, self).__init__()
        self.threshold = threshold
        self.leak = leak
        self.potential = None

    def forward(self, potential_update):
        """
        İleri Yayılım Fonksiyonu
        
        Gelen potansiyel güncellemesini kullanarak LIF nöronunun durumunu günceller
        ve aksiyon potansiyeli (spike) üretir.
        
        İşlem Adımları:
        1. Potansiyel başlatma (ilk çağrı için) - Eğer potansiyel değeri None ise veya
           giriş boyutuyla eşleşmiyorsa, sıfır tensörü ile başlatılır
        2. Sızıntılı integrasyon: V(t) = leak * V(t-1) + I(t)
           - leak: Sızıntı faktörü (0-1 arası), önceki potansiyelin ne kadarının korunacağını belirler
           - V(t-1): Önceki zaman adımındaki membran potansiyeli
           - I(t): Anlık giriş akımı/potansiyel değişimi
        3. Spike üretimi: spike = 1 if V(t) >= threshold else 0
           - threshold: Eşik değeri, bu değerin üzerindeki potansiyeller spike üretir
        4. Reset mekanizması: V(t) = 0 if spike = 1
           - Spike üretildikten sonra membran potansiyeli sıfırlanır (refrakter periyot)
        
        Parametreler:
        - potential_update: Nörona gelen giriş akımı/potansiyel değişimi
        
        Dönüş:
        - spike: Üretilen aksiyon potansiyeli (0 veya 1)
        """
        if self.potential is None or self.potential.shape != potential_update.shape:
            self.potential = torch.zeros_like(potential_update)

        self.potential = self.potential * self.leak + potential_update
        spike = (self.potential >= self.threshold).float()
        self.potential = torch.where(spike > 0, torch.zeros_like(self.potential), self.potential)
        return spike

class LayerNorm(nn.Module):
    """
    Katman Normalizasyonu
    
    Girdi aktivasyonlarını normalize eder. Her özellik boyutu için ayrı 
    ortalama ve standart sapma hesaplanır.
    
    Matematiksel işlem:
    y = gamma * (x - mean) / (std + eps) + beta
    
    Parametreler:
    - features: Normalize edilecek özellik sayısı
    - eps: Sayısal kararlılık için eklenen küçük sabit
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        İleri Yayılım Fonksiyonu
        
        Girdi aktivasyonlarını normalize eder.
        
        İşlem Adımları:
        1. Ortalama hesaplama (her örnek için)
        2. Standart sapma hesaplama
        3. Normalizasyon: (x - mean) / (std + eps)
        4. Ölçekleme ve kaydırma: gamma * norm + beta
        
        Parametreler:
        - x: Normalize edilecek girdi tensörü
        
        Dönüş:
        - normalized: Normalize edilmiş çıktı
        """
        mean = x.mean(dim=-1, keepdim=True)  # Her örnek için ortalama
        std = x.std(dim=-1, keepdim=True)    # Her örnek için standart sapma
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class PredictiveCodingLayer(nn.Module):
    """
    Öngörücü Kodlama Katmanı
    
    Beyin'in çalışma prensiplerinden esinlenen, yukarıdan-aşağı ve aşağıdan-yukarı
    bilgi akışını dengeleyen bir yapay sinir ağı katmanı.
    
    Temel bileşenler:
    1. Uyarıcı (Excitatory) nöronlar: Pozitif aktivasyon yayan nöronlar
    2. Önleyici (Inhibitory) nöronlar: Negatif aktivasyon yayan nöronlar
    3. Hata birimleri: Tahmin hatalarını hesaplayan birimler
    
    Her bir adımda:
    1. Giriş verisi ile yeniden oluşturulan giriş arasındaki hata hesaplanır
    2. Uyarıcı nöronlar güncellenir: dμ_exc = K(error) + W_ee(μ_exc) - W_ie(μ_inh)
    3. Önleyici nöronlar güncellenir: dμ_inh = W_ei(μ_exc) - W_ii(μ_inh)
    
    Parametreler:
    - input_dim: Giriş boyutu
    - units: Toplam nöron sayısı
    - recurrent_steps: Tekrarlı işlem adım sayısı
    - local_lr: Yerel öğrenme oranı
    - weight_decay: Ağırlık sönümlemesi katsayısı
    - num_levels: Hiyerarşik seviye sayısı
    - inh_ratio: Önleyici nöronların oranı
    """
    def __init__(self, input_dim, units, recurrent_steps=3, local_lr=0.0005, weight_decay=0.0, num_levels=2, inh_ratio=0.25):
        super(PredictiveCodingLayer, self).__init__()
        self.units = units
        self.recurrent_steps = recurrent_steps
        self.local_lr = local_lr
        self.weight_decay = weight_decay #Not used with local update
        self.num_levels = num_levels
        self.eps = 1e-6
        self.inh_ratio = inh_ratio #Ratio of inhibitory neurons
        self.num_inh = max(1, int(units * inh_ratio)) # Number of inhibitory units
        self.num_exc = units - self.num_inh   # Number of excitatory units


        # --- Excitatory Connections ---
        self.kernel = nn.Linear(input_dim, self.num_exc, bias=False) # Input to Excitatory
        self.recurrent_exc = nn.Linear(self.num_exc, self.num_exc, bias=False) # Recurrent Excitatory
        self.feedback_exc = nn.Linear(self.num_exc, input_dim, bias=False)   # Excitatory Feedback
        self.latent_projection = nn.Linear(input_dim, self.num_exc, bias=False) # For hierarchical processing

        # --- Inhibitory Connections ---
        self.exc_to_inh = nn.Linear(self.num_exc, self.num_inh, bias=False)  # Excitatory to Inhibitory
        self.inh_to_exc = nn.Linear(self.num_inh, self.num_exc, bias=False)  # Inhibitory to Excitatory
        self.inh_to_inh = nn.Linear(self.num_inh, self.num_inh, bias=False)  # Inhibitory to Inhibitory

        # --- Error Units ---
        self.error_units = nn.Linear(input_dim, input_dim, bias=False)


        self.lif_exc = LIFActivation()  # LIF for excitatory neurons
        self.lif_inh = LIFActivation()  # LIF for inhibitory neurons
        # self.optimizer = optim.Adam(self.parameters(), lr=local_lr, weight_decay=weight_decay) # Removed: Local update
        self.mu_exc_history = []
        self.mu_inh_history = []
        self.error_history = []

    def forward(self, inputs):
      """
      PredictiveCoding İleri Yayılım
      
      Giriş verisini öngörücü kodlama katmanından geçirir.
      
      İşlem Adımları:
      1. Başlangıç durumlarının ayarlanması:
         - Uyarıcı (excitatory) nöronlar için mu_exc
         - Önleyici (inhibitory) nöronlar için mu_inh
         - Hata birimleri için error
      
      2. Her tekrar adımında:
         a) Hata hesaplama
            - Giriş ile yeniden oluşturulan giriş arasındaki fark
            - Hata birimlerinden geçirme
            
         b) Uyarıcı nöron güncellemesi
            - Hata tabanlı güncelleme
            - Tekrarlayan bağlantılardan gelen etki
            - Önleyici nöronlardan gelen inhibisyon
            
         c) Önleyici nöron güncellemesi
            - Uyarıcı nöronlardan gelen uyarım
            - Önleyici nöronlardan gelen inhibisyon
            
      3. Geribildirim işlemi
         - Uyarıcı nöronlardan gelen geribildirim
         - Gizli temsil projeksiyonu
         - Hata hesaplama
      
      4. Spike üretimi
         - Uyarıcı nöronlar için LIF aktivasyonu
         - Önleyici nöronlar için LIF aktivasyonu
      
      Parametreler:
      - inputs: Giriş verisi [batch_size, input_dim]
      
      Dönüş:
      - spike_exc: Uyarıcı nöronların ürettiği spike'lar [batch_size, num_exc]
      """
      batch_size = inputs.size(0)
      # Initialize mu (for both excitatory and inhibitory) and error units
      mu_exc = torch.zeros((batch_size, self.num_exc), device=inputs.device, requires_grad=False)  # Excitatory mu, no grad
      mu_inh = torch.zeros((batch_size, self.num_inh), device=inputs.device, requires_grad=False)  # Inhibitory mu, no grad
      error = torch.zeros_like(inputs, device=inputs.device, requires_grad=False) # Error units

      self.mu_exc_history = []
      self.mu_inh_history = []
      self.error_history = []

      for _ in range(self.recurrent_steps):
          # --- Error Calculation (Separate Error Units) ---
          reconstructed_input = self.feedback_exc(mu_exc)
          error_update = inputs - reconstructed_input
          error = self.error_units(error_update)  # Apply error unit dynamics (can add a nonlinearity here)
          self.error_history.append(error.clone().detach())

          # --- Excitatory Neuron Update ---
          precision_error = error # No precision weighting

          # projected_error = self.kernel(precision_error) # Moved to update_weights
          d_mu_exc = self.kernel(precision_error)  + self.recurrent_exc(mu_exc) - self.inh_to_exc(mu_inh) # Recurrent and inhibitory input
          mu_exc = mu_exc + self.local_lr * d_mu_exc
          self.mu_exc_history.append(mu_exc.clone().detach())

          # --- Inhibitory Neuron Update ---
          d_mu_inh = self.exc_to_inh(mu_exc) - self.inh_to_inh(mu_inh) # Excitation from exc, inhibition from inh
          mu_inh = mu_inh + self.local_lr * d_mu_inh
          self.mu_inh_history.append(mu_inh.clone().detach())

      # --- Feedback (from Excitatory Neurons) ---
      feedback_sum = torch.zeros_like(mu_exc)

      fb = torch.tanh(self.feedback_exc(mu_exc))
      fb = self.latent_projection(fb)
      fb_error = mu_exc - fb

      # No precision weighting applied in the final feedback step
      feedback_sum += fb

      spike_exc = self.lif_exc(mu_exc + feedback_sum) # Excitatory spikes
      spike_inh = self.lif_inh(mu_inh)                # Inhibitory spikes (not used in output, but for internal dynamics)

      return spike_exc  # Output shape: (batch_size, num_exc)


    def update_weights(self, inputs, mu_exc_history, error_history):
      # self.optimizer.zero_grad() # Removed optimizer.zero_grad()
      # total_loss = 0 # Removed loss calculation

      for t in range(len(mu_exc_history)):
          mu_exc = mu_exc_history[t]
          error = error_history[t]
          reconstructed_input = self.feedback_exc(mu_exc)

          # --- Local Weight Updates (using error, pre, and post) ---

          # Update kernel (input to excitatory):  error * input^T
          delta_kernel = self.local_lr * torch.bmm(error.unsqueeze(2), inputs.unsqueeze(1)).mean(0)
          self.kernel.weight.data += delta_kernel

          # Update recurrent_exc (excitatory to excitatory): mu_exc * mu_exc^T
          delta_recurrent_exc = self.local_lr * torch.bmm(mu_exc.unsqueeze(2), mu_exc.unsqueeze(1)).mean(0)
          self.recurrent_exc.weight.data += delta_recurrent_exc

          # Update feedback_exc (excitatory to input):  input * mu_exc^T
          delta_feedback_exc = self.local_lr * torch.bmm( inputs.unsqueeze(2), mu_exc.unsqueeze(1)).mean(0)
          self.feedback_exc.weight.data += delta_feedback_exc

          # Update exc_to_inh (excitatory to inhibitory): mu_inh * mu_exc^T
          if t < len(self.mu_inh_history): # Ensure we have corresponding inhibitory activity
            mu_inh = self.mu_inh_history[t]
            delta_exc_to_inh = self.local_lr * torch.bmm(mu_inh.unsqueeze(2), mu_exc.unsqueeze(1)).mean(0)
            self.exc_to_inh.weight.data += delta_exc_to_inh

            # Update inh_to_exc (inhibitory to excitatory): - mu_exc * mu_inh^T  (negative for inhibition)
            delta_inh_to_exc = -self.local_lr * torch.bmm(mu_exc.unsqueeze(2), mu_inh.unsqueeze(1)).mean(0)
            self.inh_to_exc.weight.data += delta_inh_to_exc

            # Update inh_to_inh (inhibitory to inhibitory) : - mu_inh * mu_inh^T
            delta_inh_to_inh = -self.local_lr * torch.bmm(mu_inh.unsqueeze(2), mu_inh.unsqueeze(1)).mean(0)
            self.inh_to_inh.weight.data += delta_inh_to_inh


          # --- Update for Error Units (simple Hebbian)---
          delta_error_units = self.local_lr * torch.bmm(error.unsqueeze(2), (inputs - reconstructed_input).unsqueeze(1)).mean(0)
          self.error_units.weight.data += delta_error_units


      # total_loss.backward() # Removed .backward()
      # self.optimizer.step() # Removed optimizer.step()
      # return total_loss.item() # Removed loss return
      return 0 # Return a dummy value

class SparseHebbianLayer(nn.Module):
    """
    Seyrek Hebbian Öğrenme Katmanı
    
    Hebbian öğrenme kuralını ("Together fire, together wire") uygulayan ve
    seyrek aktivasyonları teşvik eden katman.
    
    Matematiksel işlemler:
    1. İleri yayılım: y = Wx
    2. Seyrekleştirme: y = y * (y >= top_k_threshold)
    3. Hebbian güncelleme: ΔW = α * (y * x^T)
    
    Parametreler:
    - input_dim: Giriş boyutu
    - units: Çıkış nöron sayısı
    - alpha: Hebbian öğrenme oranı
    - sparsity: Seyreklik oranı (aktif kalacak nöronların oranı)
    """
    def __init__(self, input_dim, units, alpha=0.1, sparsity=0.2):
        super(SparseHebbianLayer, self).__init__()
        self.units = units
        self.alpha = alpha
        self.sparsity = sparsity
        self.kernel = nn.Linear(input_dim, units)

    def forward(self, inputs):
        """
        İleri Yayılım ve Seyrekleştirme İşlemi
        
        Giriş verilerini işleyerek seyrek aktivasyonları oluşturur.
        
        İşlem Adımları:
        1. Doğrusal dönüşüm: activations = Wx + b
           - W: Ağırlık matrisi
           - x: Giriş vektörü
           - b: Bias terimi (varsa)
        
        2. Top-k seyrekleştirme:
           - k = units * sparsity (aktif kalacak nöron sayısı)
           - En yüksek k adet aktivasyon değeri seçilir
           - Diğer aktivasyonlar sıfırlanır
           
        3. Maske oluşturma:
           - Seçilen k adet nöron için 1
           - Diğer nöronlar için 0
           
        4. Maskeleme:
           - Aktivasyonlar maske ile çarpılır
           - Sadece en yüksek k adet aktivasyon korunur
        """
        activations = self.kernel(inputs)
        k = max(1, int(self.units * self.sparsity))
        top_k_values, _ = torch.topk(activations, k, dim=-1)
        sparse_mask = (activations >= top_k_values[:, -1, None]).float()
        activations = activations * sparse_mask
        return activations # Only return activations

    def hebbian_update(self, pre, post):
        """
        Hebbian Öğrenme Kuralı Güncellemesi
        
        "Together fire, together wire" (Birlikte ateşle, birlikte bağlan) prensibine
        dayalı olarak ağırlıkları günceller. Öncül (pre) ve ardıl (post) nöronlar
        arasındaki bağlantıları güçlendirir.
        
        Matematiksel işlemler:
        1. Dış çarpım hesaplama: post_i * pre_j
        2. Batch ortalaması alma
        3. Ağırlık güncelleme: W += alpha * (post * pre^T)
        
        Parametreler:
        - pre: Öncül nöron aktivasyonları [batch_size, input_dim]
        - post: Ardıl nöron aktivasyonları [batch_size, output_dim]
        
        İşlem detayları:
        1. Her örnek için ayrı dış çarpım hesaplanır
        2. Tüm örneklerin ortalaması alınır
        3. Ağırlıklar öğrenme oranı (alpha) ile ölçeklenmiş değerle güncellenir
        """
        batch_size = pre.size(0)
        delta_w = self.alpha * torch.bmm(
            post.unsqueeze(2),  # [batch_size, output_dim, 1]
            pre.unsqueeze(1)    # [batch_size, 1, input_dim]
        ).mean(0)  # Average over batch
        self.kernel.weight.data += delta_w

    def reset_weights(self):
        """
        Ağırlıkları Sıfırlama Fonksiyonu
        
        Xavier/Glorot başlatması kullanarak ağırlıkları yeniden başlatır.
        Bu başlatma yöntemi, girdi ve çıktı boyutlarına göre uygun
        bir ağırlık dağılımı sağlar.
        
        İşlem:
        1. Kernel ağırlıkları Xavier uniform dağılımı ile başlatılır
        2. Eğer bias varsa sıfır ile başlatılır
        """
        nn.init.xavier_uniform_(self.kernel.weight)
        if self.kernel.bias is not None:
            nn.init.zeros_(self.kernel.bias)

class NonHebbianLayer(nn.Module):
    """
    Hebbian-Olmayan Katman
    
    Standart yapay sinir ağı katmanı, ancak ağırlıklara zamanla
    sönümleme uygular.
    
    Matematiksel işlemler:
    1. İleri yayılım: y = Wx
    2. Ağırlık sönümlemesi: W = W * (1 - decay_rate)
    
    Parametreler:
    - input_dim: Giriş boyutu
    - units: Çıkış nöron sayısı
    - decay_rate: Ağırlık sönümleme oranı
    """
    def __init__(self, input_dim, units, decay_rate=0.01):
        super(NonHebbianLayer, self).__init__()
        self.units = units
        self.decay_rate = decay_rate
        self.kernel = nn.Linear(input_dim, units)

    def forward(self, inputs):
        """
        İleri Yayılım Fonksiyonu
        
        Giriş verilerini doğrusal katmandan geçirir.
        
        Matematiksel işlem:
        y = Wx + b
        - W: Ağırlık matrisi
        - x: Giriş vektörü
        - b: Bias terimi (varsa)
        
        Parametreler:
        - inputs: Giriş tensörü [batch_size, input_dim]
        
        Dönüş:
        - activations: Doğrusal dönüşüm sonucu [batch_size, units]
        """
        activations = self.kernel(inputs)
        return activations

    def non_hebbian_decay(self):
        """
        Hebbian-Olmayan Ağırlık Sönümlemesi
        
        Ağırlıkları belirli bir oranda azaltarak öğrenmeyi
        regülarize eder. Bu işlem, çok büyük ağırlıkların
        oluşmasını engeller ve modelin genelleme yeteneğini artırır.
        
        İşlem:
        W_new = W_old * (1 - decay_rate)
        """
        self.kernel.weight.data *= (1 - self.decay_rate)

class HierarchicalPredictiveCodingLayer(nn.Module):
    """
    Hiyerarşik Öngörücü Kodlama Katmanı
    
    Çoklu seviyeli öngörücü kodlama katmanlarını birleştiren yapı.
    Her seviye, bir önceki seviyenin çıktısını girdi olarak kullanır.
    
    Yapı:
    Level 1: input_dim -> input_dim/2
    Level 2: input_dim/2 -> input_dim/4
    ...
    Level n: input_dim/2^(n-1) -> units
    
    Her seviye kendi öngörücü kodlama mekanizmasına sahiptir.
    
    Parametreler:
    - input_dim: Giriş boyutu
    - units: Son seviyenin çıkış boyutu
    - num_levels: Hiyerarşi seviyesi sayısı
    - recurrent_steps: Her seviyedeki tekrarlı işlem adım sayısı
    - inh_ratio: Her seviyedeki önleyici nöronların oranı
    """
    def __init__(self, input_dim, units, num_levels=2, recurrent_steps=3, inh_ratio=0.25):
        super(HierarchicalPredictiveCodingLayer, self).__init__()
        self.units = units
        self.num_levels = num_levels
        self.recurrent_steps = recurrent_steps


        level_dims = []
        current_dim = input_dim
        for i in range(num_levels):
            level_dims.append(current_dim)
            current_dim = current_dim // 2 # Integer division for consistent dimensions

        self.levels = nn.ModuleList([
            PredictiveCodingLayer(
                input_dim=level_dims[i],
                units=level_dims[i + 1] if i < len(level_dims) - 1 else units,
                recurrent_steps=recurrent_steps,
                inh_ratio=inh_ratio  # Pass inh_ratio to each layer
            ) for i in range(num_levels)
        ])
        # self.optimizer = optim.Adam(self.parameters(), lr=0.001) # Removed: Local update
        self.all_mu_exc_histories = [] # Store histories for all levels
        self.all_error_histories = []

    def forward(self, inputs):
        """
        Hiyerarşik İleri Yayılım
        
        Girişi birden fazla öngörücü kodlama katmanından geçirir.
        Her katmanın çıktısı bir sonraki katmanın girişi olur.
        
        İşlem Adımları:
        1. Her seviye için:
           - Öngörücü kodlama işlemi
           - Spike üretimi
           - ReLU aktivasyonu
        2. Tüm seviyelerin spike'larının ortalaması alınır
        
        Matematiksel işlemler:
        - Her seviye: y_i = PC_i(x_i)
        - Seviyeler arası: x_(i+1) = ReLU(y_i)
        - Son çıktı: y = mean(y_1, y_2, ..., y_n)
        
        Parametreler:
        - inputs: İlk seviye için giriş verisi
        
        Dönüş:
        - combined_output: Tüm seviyelerin ortalama spike çıktısı
        """
        all_spikes = []
        current_input = inputs
        self.all_mu_exc_histories = []
        self.all_error_histories = []

        for level in self.levels:
            spikes = level(current_input)
            all_spikes.append(spikes)
            current_input = F.relu(spikes)  # Apply ReLU between levels
            self.all_mu_exc_histories.append(level.mu_exc_history)
            self.all_error_histories.append(level.error_history)


        combined_output = torch.stack(all_spikes).mean(dim=0)  # Average spikes from all levels
        return combined_output


    def update_weights(self, inputs): # all_mu_histories removed
        """
        Hiyerarşik Ağırlık Güncelleme
        
        Her seviyedeki öngörücü kodlama katmanının ağırlıklarını günceller.
        Her seviye, bir önceki seviyenin çıktısını kullanarak kendi ağırlıklarını
        yerel olarak günceller.
        
        İşlem Adımları:
        1. İlk seviye için giriş verisi kullanılır
        2. Her seviye için:
           - Öngörücü kodlama ağırlık güncellemesi yapılır
           - Son uyarıcı (excitatory) aktivasyonlar bir sonraki seviye için giriş olur
           - Hata değeri toplanır
        
        Parametreler:
        - inputs: İlk seviye için giriş verisi
        
        Dönüş:
        - total_loss: Tüm seviyelerin toplam hata değeri
        """
        total_loss = 0
        current_input = inputs
        for i, level in enumerate(self.levels):
            # Pass the correct mu_history and error_history to each level's update_weights
            level_loss = level.update_weights(current_input, self.all_mu_exc_histories[i], self.all_error_histories[i])
            current_input =  self.all_mu_exc_histories[i][-1].detach() # Use the last mu_exc as input to the next level
            total_loss += level_loss
        return total_loss


class ContextualOutputHead(nn.Module):
    """
    Bağlamsal Çıktı Başlığı
    
    Son katman olarak görev yapan, önceki katmanlardan gelen
    bilgiyi sınıflandırma görevine uygun formata dönüştüren katman.
    
    Yapı:
    1. Giriş -> Gizli katman (Dense + LayerNorm + ReLU + Dropout)
    2. Gizli katman -> Çıkış (Dense)
    
    Parametreler:
    - input_dim: Giriş boyutu
    - output_dim: Çıkış boyutu (sınıf sayısı)
    """
    def __init__(self, input_dim, output_dim):
        super(ContextualOutputHead, self).__init__()
        hidden_dim = max(input_dim, output_dim * 4)  # Ensure enough capacity
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        """
        Bağlamsal İleri Yayılım
        
        Giriş verilerini son sınıflandırma katmanından geçirir.
        
        İşlem Adımları:
        1. Giriş -> Yoğun katman (Dense1)
        2. Katman normalizasyonu
        3. ReLU aktivasyonu
        4. Dropout (Seyreltme)
        5. Son yoğun katman (Dense2)
        
        Matematiksel işlemler:
        1. h = Dense1(x)
        2. h = LayerNorm(h)
        3. h = ReLU(h)
        4. h = Dropout(h, p=0.1)
        5. y = Dense2(h)
        
        Parametreler:
        - inputs: Giriş tensörü [batch_size, input_dim]
        
        Dönüş:
        - outputs: Sınıf olasılıkları [batch_size, output_dim]
        """
        x = self.dense1(inputs)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

def build_model(input_dim, output_dim):
    """
    Model Oluşturma Fonksiyonu
    
    Tüm katmanları birleştirerek tam modeli oluşturur.
    
    Yapı:
    1. Öngörücü Kodlama Katmanı (input -> 256)
    2. Seyrek Hebbian Katmanı (256 -> 128)
    3. Hebbian Olmayan Katman (128 -> 27)
    4. Hiyerarşik Öngörücü Kodlama (27 -> 27/8)
    5. Bağlamsal Çıktı Başlığı (27/8 -> output)
    """
    pc_dim = 256
    model = nn.Sequential(
        PredictiveCodingLayer(input_dim, pc_dim, recurrent_steps=2, inh_ratio=0.0039),  # inh_ratio added
        SparseHebbianLayer(pc_dim - 1, (pc_dim - 1) // 2, sparsity=0.3),
        NonHebbianLayer((pc_dim - 1) // 2, 27, decay_rate=0.005),
        HierarchicalPredictiveCodingLayer(27, 27 // 8, num_levels=2, recurrent_steps=2, inh_ratio=0.25), # inh_ratio added
        ContextualOutputHead(27 // 8, output_dim)
    )
    return model

# --- Dataset and DataLoader ---
def prepare_data(dataset_name='MNIST', batch_size=64):
    """
    Veri Seti Hazırlama Fonksiyonu
    
    Seçilen veri setini yükler, önişler ve DataLoader'ları oluşturur.
    
    Desteklenen veri setleri:
    - MNIST: El yazısı rakamlar (28x28, gri tonlamalı)
    - CIFAR10: Renkli nesneler (32x32x3, RGB)
    - FashionMNIST: Giysi resimleri (28x28, gri tonlamalı)
    - Synthetic: Yapay veri (50 boyutlu rastgele vektörler)
    
    Parametreler:
    - dataset_name: Veri seti adı
    - batch_size: Mini-batch boyutu
    """
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = 28 * 28
        output_dim = 10

    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_dim = 32 * 32 * 3
        output_dim = 10
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = 28 * 28
        output_dim = 10

    elif dataset_name == "synthetic":
        num_samples = 10000
        input_dim = 50
        output_dim = 5
        data = torch.randn(num_samples, input_dim)
        targets = torch.randint(0, output_dim, (num_samples,))
        train_dataset = TensorDataset(data, targets)
        test_dataset = TensorDataset(data, targets)
    else:
        raise ValueError("Unsupported dataset name")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, input_dim, output_dim

# --- Training Loop ---
def train(model, data_loader, epochs=10, device='cpu'):
    """
    Model Eğitim Fonksiyonu
    
    Modeli verilen veri seti üzerinde eğitir. Her katman kendi özel öğrenme 
    mekanizmasını kullanarak güncellenir.
    
    Eğitim sürecindeki matematiksel işlemler:
    
    1. İleri Yayılım (Forward Pass):
       - Giriş -> PC Katmanı: Öngörücü kodlama, hata hesaplama
       - PC -> SHL: Seyrek Hebbian öğrenme, aktivasyon seyrekleştirme
       - SHL -> NHL: Hebbian olmayan işlem, ağırlık sönümlemesi
       - NHL -> HPC: Hiyerarşik öngörücü kodlama
       - HPC -> Çıktı: Son katman sınıflandırma
    
    2. Özel Katman Güncellemeleri:
       a) PredictiveCoding Katmanı:
          - Hata = Hedef - Tahmin
          - Ağırlık güncelleme: W += lr * (hata * giriş^T)
       
       b) SparseHebbian Katmanı:
          - Hebbian kuralı: W += alpha * (post * pre^T)
          - post: Çıktı aktivasyonları
          - pre: Giriş aktivasyonları
       
       c) NonHebbian Katmanı:
          - Sönümleme: W = W * (1 - decay_rate)
       
       d) HierarchicalPC Katmanı:
          - Her seviye için öngörücü kodlama güncellemesi
    
    3. Çıktı Katmanı:
       - CrossEntropy kaybı hesaplama
       - Gradyan hesaplama ve güncelleme
    
    Parametreler:
    - model: Eğitilecek model
    - data_loader: Eğitim veri yükleyicisi
    - epochs: Toplam epoch sayısı
    - device: Hesaplama cihazı (CPU/GPU)
    
    Dönüş:
    - losses: Her epoch için ortalama kayıp değerleri
    """
    """
    Model Eğitim Fonksiyonu
    
    Modeli verilen veri seti üzerinde eğitir. Her katman
    kendi özel öğrenme mekanizmasını kullanır.
    
    Eğitim adımları:
    1. İleri yayılım: Tüm katmanlardan geçiş
    2. Çıktı katmanı için CrossEntropy kaybı
    3. Her katmanın kendi özel güncellemesi:
       - PredictiveCoding: Hata-tabanlı güncelleme
       - Hebbian: Hebbian öğrenme kuralı
       - NonHebbian: Ağırlık sönümlemesi
       - HierarchicalPC: Çok seviyeli güncelleme
    
    Parametreler:
    - model: Eğitilecek model
    - data_loader: Eğitim veri yükleyicisi
    - epochs: Toplam epoch sayısı
    - device: Hesaplama cihazı (CPU/GPU)
    """
    model.to(device)
    # Initialize model weights (important for custom layers)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            if fan_in + fan_out > 0:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                print(f"Skipping initialization for layer {module} due to zero fan_in and fan_out.")
        elif isinstance(module, SparseHebbianLayer):
            module.reset_weights()

    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Removed global optimizer
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True) # Removed scheduler
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            # --- Forward pass through all layers ---
            pc_output = model[0](data)
            shl_output = model[1](pc_output)
            nhl_output = model[2](shl_output)
            hpc_output = model[3](nhl_output)
            final_output = model[4](hpc_output)

            # --- Loss Calculation (still using CrossEntropy for the output layer) ---
            loss = F.cross_entropy(final_output, target) # Keep CrossEntropy for output layer
            total_loss += loss.item()
            batch_count += 1

            # --- Local Updates (for Predictive and Hebbian Layers) ---

            # Update PredictiveCodingLayer
            model[0].update_weights(data, model[0].mu_exc_history, model[0].error_history) # Pass appropriate histories

            # Update SparseHebbianLayer
            model[1].hebbian_update(pc_output.detach(), shl_output.detach())

            # Update NonHebbianLayer
            model[2].non_hebbian_decay()

            # Update HierarchicalPredictiveCodingLayer
            model[3].update_weights(nhl_output) # mu histories are now handled inside the layer


            # --- Backpropagation ONLY for the output layer ---
            final_output.backward(torch.nn.functional.one_hot(target, num_classes=final_output.shape[1]).float()) # One-hot encoding


            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        avg_loss = total_loss / batch_count
        losses.append(avg_loss)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

        # scheduler.step(avg_loss) # Removed scheduler

    return losses

# --- Evaluation Function ---
def evaluate(model, data_loader, device='cpu'):
    """
    Model Değerlendirme Fonksiyonu
    
    Test veri seti üzerinde modelin performansını ölçer.
    Doğruluk oranını (accuracy) hesaplar ve raporlar.
    
    Parametreler:
    - model: Değerlendirilecek model
    - data_loader: Test veri yükleyicisi
    - device: Hesaplama cihazı (CPU/GPU)
    
    Dönüş:
    - accuracy: Doğruluk oranı (%)
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            pc_output = model[0](data)
            shl_output = model[1](pc_output)
            nhl_output = model[2](shl_output)
            hpc_output = model[3](nhl_output)
            final_output = model[4](hpc_output)
            _, predicted = torch.max(final_output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    """
    Ana Yürütme Modülü
    
    Model eğitimi ve değerlendirmesi için ana akış kontrolü:
    
    1. Cihaz Seçimi ve Başlatma:
       - MPS (Metal Performance Shaders, Apple)
       - CUDA (NVIDIA GPU)
       - CPU
       Her cihaz için deterministik sonuçlar için tohum (seed) ayarlanır
    
    2. Model ve Veri Hazırlığı:
       - MNIST veri seti yüklenir ve önişlenir
       - Model mimarisi oluşturulur
       - Model seçilen cihaza taşınır
    
    3. Eğitim ve Değerlendirme:
       - Model belirlenen epoch sayısı kadar eğitilir
       - Test veri seti üzerinde doğruluk oranı hesaplanır
       - Eğitilmiş model kaydedilir
    """
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset_name = 'MNIST'
    batch_size = 256
    epochs = 15

    print(f"Preparing {dataset_name} dataset...")
    train_loader, test_loader, input_dim, output_dim = prepare_data(dataset_name, batch_size)

    print("Building model...")
    model = build_model(input_dim, output_dim)
    model.to(device)

    print("Starting training...")
    losses = train(model, train_loader, epochs, device)

    print("\nEvaluating model...")
    accuracy = evaluate(model, test_loader, device)
    print("Saving model...")
    torch.save(model.state_dict(), f"{dataset_name}_model_improved.pth")
    print(f"Model saved as {dataset_name}_model_improved.pth")
