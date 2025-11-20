import sqlite3
from datetime import datetime, timedelta

db = sqlite3.connect('MONITORING/evidently/data/predictions.db')
cursor = db.cursor()

print('=== DATABASE STATUS ===')

# Total predictions
cursor.execute('SELECT COUNT(*) FROM titanic_predictions')
total_titanic = cursor.fetchone()[0]
cursor.execute('SELECT COUNT(*) FROM unemployment_predictions')
total_unemp = cursor.fetchone()[0]

print(f'Total Titanic: {total_titanic}')
print(f'Total Unemployment: {total_unemp}')

# Dernières prédictions Titanic
print('\n=== 10 DERNIERES PREDICTIONS TITANIC ===')
cursor.execute('''
    SELECT id, timestamp, model_type, sex, pclass, age, prediction
    FROM titanic_predictions
    ORDER BY timestamp DESC
    LIMIT 10
''')
print('ID | Timestamp           | Model | Sex | Pclass | Age | Pred')
print('-' * 70)
for row in cursor.fetchall():
    print(f'{row[0]:3} | {row[1]} | {row[2]:5} | {row[3]:3} | {row[4]:6} | {row[5]:4.0f} | {row[6]}')

# Distribution sex dans les dernières prédictions
print('\n=== DISTRIBUTION SEX (toutes prédictions) ===')
cursor.execute('SELECT sex, COUNT(*) FROM titanic_predictions GROUP BY sex')
for sex, count in cursor.fetchall():
    label = 'Femme' if sex == 1 else 'Homme'
    print(f'{label} (sex={sex}): {count}')

# Distribution pclass
print('\n=== DISTRIBUTION PCLASS (toutes prédictions) ===')
cursor.execute('SELECT pclass, COUNT(*) FROM titanic_predictions GROUP BY pclass')
for pclass, count in cursor.fetchall():
    print(f'Classe {pclass}: {count}')

# Prédictions des dernières 30 minutes
print('\n=== PREDICTIONS DERNIERES 30 MINUTES ===')
cutoff = (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
cursor.execute(f'''
    SELECT COUNT(*), model_type, AVG(sex), AVG(pclass)
    FROM titanic_predictions
    WHERE timestamp >= '{cutoff}'
    GROUP BY model_type
''')
recent = cursor.fetchall()
if recent:
    for count, model, avg_sex, avg_pclass in recent:
        print(f'{model}: {count} predictions | avg_sex={avg_sex:.2f} | avg_pclass={avg_pclass:.2f}')
else:
    print('Aucune prediction recente dans les 30 dernieres minutes!')

# Prédictions dernière heure
print('\n=== PREDICTIONS DERNIERE HEURE ===')
cutoff_1h = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
cursor.execute(f'''
    SELECT COUNT(*), model_type
    FROM titanic_predictions
    WHERE timestamp >= '{cutoff_1h}'
    GROUP BY model_type
''')
recent_1h = cursor.fetchall()
if recent_1h:
    for count, model in recent_1h:
        print(f'{model}: {count} predictions')
else:
    print('Aucune prediction dans la derniere heure!')

db.close()
