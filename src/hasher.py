import bcrypt

# Şifreler listesi
passwords = ['123', '456']
hashed_passwords = []

print("🔐 Şifreler oluşturuluyor...")

for password in passwords:
    # Şifreyi byte formatına çevir
    pwd_bytes = password.encode('utf-8')
    # Salt oluştur ve hashle
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    # String'e çevirip listeye ekle
    hashed_passwords.append(hashed.decode('utf-8'))

print("\n✅ Şifreler hazır! Aşağıdaki satırları kopyalayıp config/secrets.yaml dosyasına yapıştır:\n")
print(f"Ahmet (123) için password: {hashed_passwords[0]}")
print(f"Mehmet (456) için password: {hashed_passwords[1]}")