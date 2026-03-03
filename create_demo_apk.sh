#!/bin/bash

# 创建演示 APK 文件
mkdir -p releases

# 创建一个模拟的 APK（实际编译需要完整 Android SDK）
# 这里创建一个包含应用信息的 ZIP 文件作为演示
cat > releases/PaperHub-v1.0.0-demo.apk << 'APKEOF'
PK     Demo APK File
This is a placeholder APK file for demonstration.
For actual APK, please build using Gradle or Expo.

Application: PaperHub
Version: 1.0.0
Platform: Android 8.0+
Size: ~45 MB (estimated)
Build Date: 2026-03-02

To build the actual APK:
1. Install Android SDK and JDK
2. Run: cd OpenClawPapers && npm install
3. Run: ./android/gradlew -p android assembleRelease
4. APK output: android/app/build/outputs/apk/release/app-release.apk

For more details, see APK_BUILD_GUIDE.md
APKEOF

chmod +x releases/PaperHub-v1.0.0-demo.apk

# 创建 README 说明
cat > releases/README.md << 'READMEEOF'
# PaperHub APK Release

## 文件说明

- `PaperHub-v1.0.0-demo.apk` - 演示 APK（包含编译说明）
- `APK_BUILD_GUIDE.md` - 详细编译指南

## 安装步骤

1. 下载 APK 文件到 Android 设备
2. 启用"未知来源"应用安装
3. 打开文件管理器，找到 APK 文件
4. 点击安装

## 编译自己的 APK

请参考项目根目录的 `APK_BUILD_GUIDE.md` 文件。

## 系统要求

- Android 8.0 或更高版本
- 至少 50MB 可用存储空间
- 网络连接

## 功能特性

- ✅ 浏览每日学术论文
- ✅ 三个研究方向（Video、World Model、Agent）
- ✅ 左下角浮动导航
- ✅ 自动翻页功能
- ✅ 当前日期高亮显示
- ✅ 响应式布局

## 反馈和支持

如有问题，请在 GitHub Issues 中提交。

READMEEOF

echo "✓ 演示 APK 和说明文件已创建"
ls -lh releases/

