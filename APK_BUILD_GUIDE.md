# PaperHub APK 编译指南

## 📋 前置要求

### 系统要求
- **操作系统** - Linux (Ubuntu 20.04+) / macOS / Windows (WSL2)
- **Node.js** - v22.11.0 或更高版本
- **npm** - v10.9.2 或更高版本
- **JDK** - OpenJDK 11 或更高版本
- **Android SDK** - API Level 31+
- **Gradle** - 7.0 或更高版本

### 环境变量设置

```bash
# 设置 JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# 设置 ANDROID_HOME
export ANDROID_HOME=$HOME/Android/Sdk

# 添加到 PATH
export PATH=$PATH:$JAVA_HOME/bin:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```

## 🚀 快速编译

### 方法 1：使用 Gradle 本地编译（推荐）

```bash
cd OpenClawPapers

# 1. 安装依赖
npm install

# 2. 编译 Release APK
./android/gradlew -p android assembleRelease

# 3. 签名 APK（可选）
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 \
  -keystore my-release-key.keystore \
  android/app/build/outputs/apk/release/app-release-unsigned.apk \
  alias_name

# 4. 对齐 APK
zipalign -v 4 app-release-unsigned.apk PaperHub-v1.0.0.apk
```

### 方法 2：使用 Expo 云编译（最简单）

```bash
# 安装 Expo CLI
npm install -g eas-cli

# 登录 Expo 账号
eas login

# 云编译 APK
eas build --platform android --non-interactive

# 编译完成后，下载链接会显示在终端
```

### 方法 3：使用 React Native CLI

```bash
cd OpenClawPapers

# 编译 Debug APK
npm run android

# 编译 Release APK
npx react-native run-android --variant=release
```

## 📦 编译输出

编译成功后，APK 文件位置：

```
OpenClawPapers/android/app/build/outputs/apk/release/app-release.apk
```

文件大小：约 40-50 MB

## 🔐 APK 签名

### 生成签名密钥

```bash
keytool -genkey -v -keystore my-release-key.keystore \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -alias my-key-alias
```

### 使用 Gradle 自动签名

在 `android/app/build.gradle` 中配置：

```gradle
android {
  ...
  signingConfigs {
    release {
      storeFile file('my-release-key.keystore')
      storePassword 'your-store-password'
      keyAlias 'my-key-alias'
      keyPassword 'your-key-password'
    }
  }
  buildTypes {
    release {
      signingConfig signingConfigs.release
    }
  }
}
```

## 📤 发布到 GitHub Release

### 1. 创建 GitHub Release

```bash
# 设置 GitHub Token
export GITHUB_TOKEN=your_github_token

# 使用 gh CLI 创建 Release
gh release create v1.0.0 \
  --title "PaperHub v1.0.0" \
  --notes "首个正式版本发布" \
  OpenClawPapers/android/app/build/outputs/apk/release/app-release.apk
```

### 2. 手动上传

1. 访问 GitHub 仓库
2. 点击 "Releases" 标签
3. 点击 "Draft a new release"
4. 填写版本号和发布说明
5. 上传 APK 文件
6. 发布

## 🐛 常见问题

### Q: 编译时出现 "JAVA_HOME not set" 错误

**解决方案：**
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
```

### Q: 编译时出现 "Android SDK not found" 错误

**解决方案：**
```bash
# 安装 Android SDK
sudo apt-get install android-sdk

# 或使用 Android Studio 下载 SDK

# 设置环境变量
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```

### Q: 编译时出现 "Gradle build failed" 错误

**解决方案：**
```bash
# 清除 Gradle 缓存
./android/gradlew clean

# 重新编译
./android/gradlew -p android assembleRelease
```

### Q: APK 文件过大

**优化方案：**
- 启用 ProGuard/R8 混淆
- 移除未使用的库
- 使用 App Bundle 分割

## 📊 编译时间

- **Debug APK** - 2-5 分钟
- **Release APK** - 5-10 分钟
- **Expo 云编译** - 10-15 分钟

## 🔗 相关资源

- [React Native 编译指南](https://reactnative.dev/docs/signed-apk-android)
- [Gradle 官方文档](https://gradle.org/releases/)
- [Android 开发者指南](https://developer.android.com/guide)
- [Expo 云编译文档](https://docs.expo.dev/build/setup/)

## 📝 版本管理

建议使用以下版本号格式：

```
vX.Y.Z-build_number

例如：
v1.0.0-001  # 首个发布版本
v1.0.1-002  # Bug 修复版本
v1.1.0-003  # 新功能版本
```

---

**最后更新** - 2026 年 3 月 2 日
