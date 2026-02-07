#!/usr/bin/env python3
"""
Анализатор событий информационной безопасности
Домашнее задание №9 - ФИНАЛЬНАЯ ВЕРСИЯ
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
import warnings
from collections import Counter
import numpy as np

warnings.filterwarnings('ignore')

class SecurityEventsAnalyzer:
    def __init__(self, json_file='events.json'):
        """
        Инициализация анализатора событий безопасности
        
        Args:
            json_file (str): Путь к файлу JSON с данными о событиях
        """
        self.json_file = json_file
        self.df = None
        self.output_dir = 'output'
        self.pattern_lengths_to_check = [3, 5, 8]  # Разные длины паттернов для анализа
        
        # Создаем директорию для результатов
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Создана директория для результатов: {self.output_dir}")
    
    def load_and_clean_data(self):
        """
        Загрузка и очистка данных из JSON файла с обработкой ошибок форматирования
        """
        try:
            # Проверяем существование файла
            if not os.path.exists(self.json_file):
                raise FileNotFoundError(f"❌ Файл {self.json_file} не найден")
            
            print(f"📂 Загрузка данных из файла: {self.json_file}")
            
            # Загружаем сырые данные JSON
            with open(self.json_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Обрабатываем возможные проблемы с форматом
            # Ищем ключ 'events' с любыми пробелами вокруг
            events_key = None
            for key in raw_data.keys():
                if key.strip().lower() == 'events':
                    events_key = key
                    break
            
            if not events_key:
                # Если ключ не найден, пытаемся найти похожий
                for key in raw_data.keys():
                    if 'event' in key.lower():
                        events_key = key
                        print(f"⚠️  Найден похожий ключ: '{key}' вместо 'events'")
                        break
            
            if not events_key:
                raise KeyError("❌ Не найден ключ с событиями в JSON файле")
            
            print(f"✅ Найден ключ событий: '{events_key}'")
            events = raw_data[events_key]
            
            # Преобразуем в DataFrame
            self.df = pd.DataFrame(events)
            
            # Очищаем имена колонок от лишних пробелов
            self.df.columns = [col.strip() for col in self.df.columns]
            
            # Очищаем данные в колонках от лишних пробелов
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].astype(str).str.strip()
            
            # Преобразуем timestamp в datetime
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
                
                # Проверяем успешность преобразования
                if self.df['timestamp'].isnull().any():
                    print("⚠️  Некоторые временные метки не удалось преобразовать")
                
                # Извлекаем дополнительные временные метрики
                self.df['date'] = self.df['timestamp'].dt.date
                self.df['hour'] = self.df['timestamp'].dt.hour
                self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
                self.df['day_of_week_num'] = self.df['timestamp'].dt.dayofweek
                self.df['date_str'] = self.df['date'].astype(str)
            
            print(f"✅ Данные успешно загружены: {len(self.df)} записей")
            print(f"📊 Колонки: {list(self.df.columns)}")
            
            # Выводим информацию о временном диапазоне
            if 'timestamp' in self.df.columns:
                print(f"📅 Временной диапазон: {self.df['timestamp'].min()} - {self.df['timestamp'].max()}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка формата JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Ошибка при загрузке данных: {type(e).__name__}: {e}")
            sys.exit(1)
    
    def extract_detailed_threat_category(self, signature):
        """
        Расширенная детализированная категоризация угроз
        
        Args:
            signature (str): Сигнатура события
        
        Returns:
            tuple: (основная_категория, детальная_категория)
        """
        sig = str(signature).upper()
        
        # Основные категории
        if 'MALWARE-CNC' in sig:
            main_cat = 'MALWARE'
            if 'WIN.TROJAN' in sig:
                detail = 'Trojan/Win.Jadtre'
            elif 'USER-AGENT' in sig:
                detail = 'C&C Communication'
            else:
                detail = 'Malware Activity'
        elif 'EXPLOIT' in sig:
            main_cat = 'EXPLOIT'
            if 'WIN32K' in sig:
                detail = 'Privilege Escalation (Win32k)'
            elif 'JAVA JRE' in sig or 'WEBLOGIC' in sig:
                detail = 'Remote Code Execution (Java)'
            elif 'ORACLE 9I' in sig:
                detail = 'Buffer Overflow (Oracle)'
            elif 'IIS' in sig:
                detail = 'Web Server Exploit'
            else:
                detail = 'Generic Exploit'
        elif 'NETBIOS' in sig:
            main_cat = 'NETWORK'
            if 'DCERPC' in sig:
                detail = 'RPC Service Exploit'
            elif 'SMB-DS' in sig:
                detail = 'SMB Service Exploit'
            else:
                detail = 'Network Protocol Anomaly'
        elif 'INDICATOR-COMPROMISE' in sig:
            main_cat = 'INDICATOR'
            if 'MYSQL' in sig:
                detail = 'Database Reconnaissance'
            else:
                detail = 'Suspicious Activity'
        elif 'RCE' in sig:
            main_cat = 'EXPLOIT'
            detail = 'Remote Code Execution'
        elif 'PRIVILEGE' in sig or 'ELEVATION' in sig:
            main_cat = 'EXPLOIT'
            detail = 'Privilege Escalation'
        elif 'BUFFER' in sig or 'BO' in sig or 'OVERFLOW' in sig:
            main_cat = 'EXPLOIT'
            detail = 'Buffer Overflow'
        else:
            main_cat = 'OTHER'
            detail = 'Uncategorized'
        
        return main_cat, detail
    
    def analyze_signature_distribution(self):
        """
        Комплексный анализ распределения событий по типам
        """
        print("\n" + "="*70)
        print("📈 КОМПЛЕКСНЫЙ АНАЛИЗ РАСПРЕДЕЛЕНИЯ СОБЫТИЙ")
        print("="*70)
        
        if 'signature' not in self.df.columns:
            print("❌ Колонка 'signature' не найдена в данных")
            return None
        
        # Общая статистика
        total_events = len(self.df)
        unique_signatures = self.df['signature'].nunique()
        
        print(f"📊 ВСЕГО СОБЫТИЙ: {total_events}")
        print(f"🔢 УНИКАЛЬНЫХ СИГНАТУР: {unique_signatures}")
        
        # Подсчет событий по сигнатурам
        signature_counts = self.df['signature'].value_counts()
        
        print(f"\n🏆 ТОП-10 НАИБОЛЕЕ ЧАСТЫХ СОБЫТИЙ:")
        for i, (signature, count) in enumerate(signature_counts.head(10).items(), 1):
            percentage = (count / total_events) * 100
            signature_display = signature[:55] + "..." if len(signature) > 55 else signature
            print(f"  {i:2}. {signature_display:60} - {count:3} ({percentage:5.1f}%)")
        
        # Детализированная категоризация угроз
        print(f"\n🎯 ДЕТАЛИЗИРОВАННАЯ КАТЕГОРИЗАЦИЯ УГРОЗ:")
        
        # Применяем расширенную категоризацию
        self.df[['threat_main_category', 'threat_detailed_category']] = self.df['signature'].apply(
            lambda x: pd.Series(self.extract_detailed_threat_category(x))
        )
        
        # Анализ по основным категориям
        main_cat_counts = self.df['threat_main_category'].value_counts()
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО ОСНОВНЫМ КАТЕГОРИЯМ:")
        for category, count in main_cat_counts.items():
            percentage = (count / total_events) * 100
            print(f"  • {category:25} - {count:3} событий ({percentage:5.1f}%)")
        
        # Анализ по детальным категориям
        detailed_cat_counts = self.df['threat_detailed_category'].value_counts()
        print(f"\n🔍 РАСПРЕДЕЛЕНИЕ ПО ДЕТАЛЬНЫМ КАТЕГОРИЯМ:")
        for category, count in detailed_cat_counts.head(8).items():
            percentage = (count / total_events) * 100
            print(f"  • {category:35} - {count:3} событий ({percentage:5.1f}%)")
        
        return signature_counts
    
    def analyze_temporal_patterns(self):
        """
        Расширенный анализ временных паттернов с поиском цикличности
        """
        print("\n" + "="*70)
        print("🕐 РАСШИРЕННЫЙ АНАЛИЗ ВРЕМЕННЫХ ПАТТЕРНОВ")
        print("="*70)
        
        if 'timestamp' not in self.df.columns:
            print("❌ Колонка 'timestamp' не найдена")
            return
        
        # Анализ по часам суток
        print(f"\n⏰ РАСПРЕДЕЛЕНИЕ ПО ЧАСАМ СУТОК:")
        
        if 'hour' in self.df.columns:
            hourly_counts = self.df['hour'].value_counts().sort_index()
            
            for hour in range(24):
                count = hourly_counts.get(hour, 0)
                if count > 0:
                    percentage = (count / len(self.df)) * 100
                    bar = "█" * int(count / max(1, hourly_counts.max() / 20))
                    print(f"  {hour:2}:00 - {hour:2}:59 | {count:3} событий | {bar} ({percentage:5.1f}%)")
            
            # Находим наиболее активные часы
            most_active_hour = hourly_counts.idxmax()
            most_active_count = hourly_counts.max()
            print(f"\n  🎯 Самый активный час: {most_active_hour}:00 ({most_active_count} событий)")
        
        # Анализ по дням
        if 'date' in self.df.columns:
            print(f"\n📅 РАСПРЕДЕЛЕНИЕ ПО ДНЯМ:")
            daily_counts = self.df['date'].value_counts().sort_index()
            
            for date, count in daily_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  • {date} - {count:3} событий ({percentage:5.1f}%)")
            
            avg_events_per_day = daily_counts.mean()
            print(f"\n  📈 Среднее количество событий в день: {avg_events_per_day:.1f}")
        
        # Поиск циклических паттернов
        print(f"\n🔄 АНАЛИЗ ЦИКЛИЧЕСКИХ ПАТТЕРНОВ:")
        
        signatures_list = self.df['signature'].tolist()
        patterns_found = False
        
        for pattern_length in self.pattern_lengths_to_check:
            print(f"\n  🔍 Проверка паттернов длиной {pattern_length} событий:")
            
            for i in range(len(signatures_list) - pattern_length * 2):
                pattern = tuple(signatures_list[i:i + pattern_length])
                next_pattern = tuple(signatures_list[i + pattern_length:i + pattern_length * 2])
                
                if pattern == next_pattern:
                    patterns_found = True
                    print(f"    ✅ Найден повторяющийся паттерн (начало: {i}):")
                    for j, sig in enumerate(pattern):
                        sig_display = sig[:50] + "..." if len(sig) > 50 else sig
                        print(f"       {j+1:2}. {sig_display}")
                    
                    # Проверяем, продолжается ли паттерн дальше
                    repetitions = 1
                    for k in range(2, 10):  # Проверяем до 10 повторений
                        check_start = i + pattern_length * k
                        if check_start + pattern_length > len(signatures_list):
                            break
                        
                        check_pattern = tuple(signatures_list[check_start:check_start + pattern_length])
                        if check_pattern == pattern:
                            repetitions += 1
                        else:
                            break
                    
                    if repetitions > 1:
                        print(f"       🔄 Паттерн повторяется {repetitions} раз подряд")
                    
                    break  # Нашли один паттерн этой длины
            
            if not patterns_found:
                print(f"    ❌ Повторяющихся паттернов длиной {pattern_length} не обнаружено")
    
    def visualize_distribution(self, signature_counts):
        """
        Создание комплекса визуализаций с сохранением в разные форматы
        """
        print("\n" + "="*70)
        print("🎨 СОЗДАНИЕ ПРОФЕССИОНАЛЬНЫХ ВИЗУАЛИЗАЦИЙ")
        print("="*70)
        
        # Настройка стилей
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Создаем все графики
        self._create_threat_category_charts()
        self._create_signature_distribution_charts(signature_counts)
        self._create_temporal_analysis_charts()
        self._create_comprehensive_heatmap()
        
        print(f"\n✅ Все графики сохранены в директории '{self.output_dir}/'")
        print("   Форматы: PNG (для просмотра) и SVG (для отчетов)")
    
    def _create_threat_category_charts(self):
        """Создание графиков по категориям угроз"""
        if 'threat_main_category' not in self.df.columns:
            return
        
        # 1. Круговая диаграмма (основные категории)
        plt.figure(figsize=(12, 10))
        threat_counts = self.df['threat_main_category'].value_counts()
        
        # Автодоля для выделения
        explode = [0.05] * len(threat_counts)
        
        wedges, texts, autotexts = plt.pie(
            threat_counts.values,
            labels=threat_counts.index,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(threat_counts.values))})',
            startangle=90,
            explode=explode,
            shadow=True,
            colors=sns.color_palette("Set3"),
            textprops={'fontsize': 10}
        )
        
        # Улучшаем отображение
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.title('Распределение событий по основным категориям угроз', 
                 fontsize=16, fontweight='bold', pad=25)
        plt.axis('equal')
        
        # Сохраняем в разных форматах
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/1_threat_categories_pie.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/1_threat_categories_pie.svg', format='svg', bbox_inches='tight')
        plt.close()
        print("   ✅ 1. Круговая диаграмма категорий угроз (PNG+SVG)")
        
        # 2. Столбчатая диаграмма (детальные категории)
        if 'threat_detailed_category' in self.df.columns:
            plt.figure(figsize=(14, 8))
            detailed_counts = self.df['threat_detailed_category'].value_counts().head(10)
            
            bars = plt.bar(
                range(len(detailed_counts)),
                detailed_counts.values,
                color=sns.color_palette("viridis", len(detailed_counts)),
                edgecolor='black',
                linewidth=0.5
            )
            
            # Добавляем значения
            for bar, count in zip(bars, detailed_counts.values):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + max(detailed_counts.values) * 0.01,
                    str(count),
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )
            
            plt.xticks(range(len(detailed_counts)), detailed_counts.index, rotation=45, ha='right')
            plt.ylabel('Количество событий', fontsize=12)
            plt.title('Топ-10 детальных категорий угроз', fontsize=16, fontweight='bold', pad=20)
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/2_detailed_categories_bar.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/2_detailed_categories_bar.svg', format='svg', bbox_inches='tight')
            plt.close()
            print("   ✅ 2. Столбчатая диаграмма детальных категорий (PNG+SVG)")
    
    def _create_signature_distribution_charts(self, signature_counts):
        """Создание графиков распределения сигнатур"""
        if signature_counts is None or len(signature_counts) == 0:
            return
        
        # Топ-15 сигнатур
        plt.figure(figsize=(16, 10))
        top_signatures = signature_counts.head(15)
        
        # Создаем горизонтальную диаграмму для лучшей читаемости
        y_pos = np.arange(len(top_signatures))
        
        bars = plt.barh(
            y_pos,
            top_signatures.values,
            color=sns.color_palette("coolwarm", len(top_signatures)),
            edgecolor='black',
            linewidth=0.5,
            height=0.7
        )
        
        # Добавляем значения и проценты
        total = signature_counts.sum()
        for i, (bar, count) in enumerate(zip(bars, top_signatures.values)):
            percentage = (count / total) * 100
            plt.text(
                count + max(top_signatures.values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count} ({percentage:.1f}%)",
                va='center',
                fontweight='bold',
                fontsize=9
            )
        
        plt.yticks(y_pos, top_signatures.index, fontsize=9)
        plt.xlabel('Количество событий', fontsize=12)
        plt.title('Топ-15 наиболее частых сигнатур безопасности', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()  # Самая частая сверху
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_top_signatures_bar.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/3_top_signatures_bar.svg', format='svg', bbox_inches='tight')
        plt.close()
        print("   ✅ 3. Диаграмма топ-15 сигнатур (PNG+SVG)")
    
    def _create_temporal_analysis_charts(self):
        """Создание графиков временного анализа"""
        if 'hour' not in self.df.columns:
            return
        
        # 1. График активности по часам
        plt.figure(figsize=(14, 7))
        
        hourly_counts = self.df['hour'].value_counts().sort_index()
        
        # Линейный график с заполнением
        plt.plot(
            hourly_counts.index,
            hourly_counts.values,
            marker='o',
            markersize=8,
            linewidth=3,
            color='#FF6B6B',
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor='#FF6B6B'
        )
        
        plt.fill_between(hourly_counts.index, hourly_counts.values, alpha=0.2, color='#FF6B6B')
        
        # Аннотации для пиков
        for hour in hourly_counts.nlargest(3).index:
            count = hourly_counts[hour]
            plt.annotate(
                f'{hour}:00\n{count}',
                xy=(hour, count),
                xytext=(hour, count + max(hourly_counts.values) * 0.05),
                ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
            )
        
        plt.xlabel('Час суток', fontsize=12)
        plt.ylabel('Количество событий', fontsize=12)
        plt.title('Суточная активность событий безопасности', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/4_hourly_activity.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/4_hourly_activity.svg', format='svg', bbox_inches='tight')
        plt.close()
        print("   ✅ 4. График суточной активности (PNG+SVG)")
    
    def _create_comprehensive_heatmap(self):
        """Создание тепловой карты активности"""
        if 'date' not in self.df.columns or 'hour' not in self.df.columns:
            return
        
        # Подготовка данных для тепловой карты
        heatmap_data = self.df.pivot_table(
            index='date_str',
            columns='hour',
            values='signature',
            aggfunc='count',
            fill_value=0
        )
        
        # Сортируем даты
        heatmap_data = heatmap_data.sort_index()
        
        plt.figure(figsize=(16, 8))
        
        # Создаем тепловую карту
        sns.heatmap(
            heatmap_data,
            cmap='YlOrRd',
            annot=True,
            fmt='g',
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Количество событий', 'shrink': 0.8},
            annot_kws={'size': 8}
        )
        
        plt.title('Тепловая карта активности событий безопасности\nДни × Часы суток', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Час суток', fontsize=12)
        plt.ylabel('Дата', fontsize=12)
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/5_activity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/5_activity_heatmap.svg', format='svg', bbox_inches='tight')
        plt.close()
        print("   ✅ 5. Тепловая карта активности (PNG+SVG)")
    
    def export_results(self):
        """
        Комплексный экспорт результатов анализа
        """
        print("\n" + "="*70)
        print("💾 КОМПЛЕКСНЫЙ ЭКСПОРТ РЕЗУЛЬТАТОВ")
        print("="*70)
        
        try:
            # 1. Экспорт полных данных в CSV
            csv_file = f'{self.output_dir}/security_events_full_data.csv'
            self.df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ 1. Полные данные экспортированы в CSV: {csv_file}")
            
            # 2. Сводная статистика
            summary_stats = {
                'Метрика': [
                    'Всего событий',
                    'Уникальных сигнатур',
                    'Период начала',
                    'Период окончания',
                    'Наиболее активный час',
                    'Среднее событий в час',
                    'Дней в данных',
                    'Основная категория угроз',
                    'Самая частая сигнатура',
                    'Коэффициент уникальности'
                ],
                'Значение': [
                    len(self.df),
                    self.df['signature'].nunique(),
                    self.df['timestamp'].min().strftime('%Y-%m-%d %H:%M') if 'timestamp' in self.df.columns else 'N/A',
                    self.df['timestamp'].max().strftime('%Y-%m-%d %H:%M') if 'timestamp' in self.df.columns else 'N/A',
                    self.df['hour'].mode()[0] if 'hour' in self.df.columns else 'N/A',
                    len(self.df) / 24 if 'hour' in self.df.columns else 'N/A',
                    self.df['date'].nunique() if 'date' in self.df.columns else 'N/A',
                    self.df['threat_main_category'].mode()[0] if 'threat_main_category' in self.df.columns else 'N/A',
                    self.df['signature'].mode()[0] if 'signature' in self.df.columns else 'N/A',
                    f"{self.df['signature'].nunique() / len(self.df):.2%}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_stats)
            summary_file = f'{self.output_dir}/summary_statistics.csv'
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"✅ 2. Сводная статистика: {summary_file}")
            
            # 3. Распределение по категориям
            if 'threat_main_category' in self.df.columns:
                category_stats = self.df.groupby('threat_main_category').agg({
                    'signature': ['count', lambda x: x.nunique()],
                    'hour': ['mean', 'std']
                }).round(2)
                
                category_stats.columns = ['Количество', 'Уникальных_сигнатур', 'Средний_час', 'Стд_час']
                category_file = f'{self.output_dir}/category_statistics.csv'
                category_stats.to_csv(category_file, encoding='utf-8-sig')
                print(f"✅ 3. Статистика по категориям: {category_file}")
            
            # 4. Экспорт в Excel (если установлен openpyxl)
            try:
                excel_file = f'{self.output_dir}/security_events_analysis.xlsx'
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    # Лист с полными данными
                    self.df.to_excel(writer, sheet_name='Все события', index=False)
                    
                    # Лист со сводной статистикой
                    summary_df.to_excel(writer, sheet_name='Сводная статистика', index=False)
                    
                    # Лист с распределением по часам
                    if 'hour' in self.df.columns:
                        hourly_stats = self.df['hour'].value_counts().sort_index()
                        hourly_stats.name = 'Количество событий'
                        hourly_stats.to_excel(writer, sheet_name='По часам')
                    
                    # Лист с топ-20 сигнатур
                    if 'signature' in self.df.columns:
                        top_signatures = self.df['signature'].value_counts().head(20)
                        top_signatures.to_excel(writer, sheet_name='Топ-20 сигнатур')
                
                print(f"✅ 4. Данные экспортированы в Excel: {excel_file}")
                
            except ImportError:
                print("ℹ️  Для экспорта в Excel установите: pip install openpyxl")
            except Exception as e:
                print(f"⚠️  Ошибка при экспорте в Excel: {e}")
            
            print(f"\n📁 Все файлы сохранены в директории: {os.path.abspath(self.output_dir)}")
            
        except Exception as e:
            print(f"❌ Ошибка при экспорте результатов: {e}")
    
    def run_full_analysis(self):
        """
        Запуск полного цикла анализа
        """
        print("\n" + "="*70)
        print("🔍 ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА СОБЫТИЙ ИБ")
        print("="*70)
        print("Версия: 2.0 | Автор: Анализатор ДЗ №9\n")
        
        # 1. Загрузка и очистка данных
        self.load_and_clean_data()
        
        # 2. Анализ распределения
        signature_counts = self.analyze_signature_distribution()
        
        # 3. Анализ временных паттернов
        self.analyze_temporal_patterns()
        
        # 4. Визуализация
        if signature_counts is not None:
            self.visualize_distribution(signature_counts)
        
        # 5. Экспорт результатов
        self.export_results()
        
        print("\n" + "="*70)
        print("🎉 АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        print("="*70)
        print(f"📊 Создано файлов: {len(os.listdir(self.output_dir)) if os.path.exists(self.output_dir) else 0}")
        print(f"📈 Проанализировано событий: {len(self.df)}")
        print(f"🔍 Обнаружено категорий угроз: {self.df['threat_main_category'].nunique() if 'threat_main_category' in self.df.columns else 0}")
        print(f"📁 Результаты: {os.path.abspath(self.output_dir)}")

def main():
    """
    Точка входа в программу
    """
    print("="*70)
    print("АНАЛИЗАТОР СОБЫТИЙ ИНФОРМАЦИОННОЙ БЕЗОПАСНОСТИ")
    print("Домашнее задание №9 - Python для аналитиков ИБ")
    print("="*70)
    
    analyzer = SecurityEventsAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
