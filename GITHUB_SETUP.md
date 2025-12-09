# 🚀 GitHub'a Yükleme Rehberi

## Adım 1: GitHub'da Yeni Repository Oluştur

1. **GitHub.com'a git** ve giriş yap
2. Sağ üstteki **"+"** butonuna tıkla → **"New repository"**
3. Repository bilgilerini doldur:
   - **Repository name**: `iot-smart-fridge` (veya istediğin isim)
   - **Description**: "IoT Smart Fridge Simulator with YOLO detection and cloud integration"
   - **Public** veya **Private** seç (öneri: Public)
   - **⚠️ ÖNEMLİ:** "Initialize this repository with a README" seçeneğini **İŞARETLEME**
   - "Add .gitignore" ve "Choose a license" seçeneklerini **boş bırak**
4. **"Create repository"** butonuna tıkla

## Adım 2: Terminal Komutları (Sırayla Çalıştır)

Aşağıdaki komutları **sırayla** terminalde çalıştır:

```bash
# 1. Proje klasörüne git
cd /Users/kadirakyurek/Desktop/internet_of_things/smart_fridge

# 2. Git repository başlat
git init

# 3. Tüm dosyaları ekle (staging area)
git add .

# 4. İlk commit'i yap
git commit -m "Initial commit: IoT Smart Fridge Simulation"

# 5. GitHub repo URL'ini ekle (YOUR_USERNAME ve REPO_NAME'i değiştir!)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 6. Ana branch'i main olarak ayarla
git branch -M main

# 7. GitHub'a yükle
git push -u origin main
```

## Adım 3: GitHub Kullanıcı Adı ve Repo İsmini Bulma

**GitHub repo URL'i şu formatta olacak:**
```
https://github.com/KULLANICI_ADIN/REPO_ISMI.git
```

**Örnek:**
- Kullanıcı adın: `kadirakyurek`
- Repo ismi: `iot-smart-fridge`
- URL: `https://github.com/kadirakyurek/iot-smart-fridge.git`

## Adım 4: Authentication (İlk Kez İse)

Eğer ilk kez GitHub'a push yapıyorsan, GitHub şifren veya **Personal Access Token** isteyebilir.

**Personal Access Token oluşturma:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" → "repo" seçeneklerini işaretle
3. Token'ı kopyala ve güvenli bir yere sakla
4. `git push` yaparken şifre yerine bu token'ı kullan

## ✅ Başarılı Olursa

Terminal'de şunu göreceksin:
```
Enumerating objects: XX, done.
Counting objects: 100% (XX/XX), done.
...
To https://github.com/YOUR_USERNAME/REPO_NAME.git
 * [new branch]      main -> main
Branch 'main' set up to track 'remote branch 'main' from 'origin'.
```

## 🔧 Sorun Giderme

**"remote origin already exists" hatası:**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

**"Authentication failed" hatası:**
- Personal Access Token kullan
- Veya GitHub Desktop uygulamasını kullan

**"Large files" hatası:**
- `models/best.pt` dosyası çok büyükse GitHub LFS kullan:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/best.pt
git commit -m "Add model file with LFS"
```

