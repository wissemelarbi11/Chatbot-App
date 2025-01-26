import nltk
import json
import random
import numpy as np
import tensorflow as tf
import tflearn
from flask import Flask, render_template, request, jsonify
import pickle
from nltk.stem.lancaster import LancasterStemmer

# Initialisation du stemmer
nltk.download('punkt')
stemmer = LancasterStemmer()

# Chargement des données d'intention
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Préparation des données de formation
words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Perform stemming and lowercasing, then remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Création des données d'entraînement
training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Création du modèle TensorFlow
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Entraînement du modèle
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# Sauvegarde des données d'entraînement
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

# Chargement du modèle et des données
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

model.load('./model.tflearn')

# Fonction pour nettoyer la phrase (tokenisation et stemming)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Fonction pour créer la bag of words
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

# Seuil d'erreur pour la classification
ERROR_THRESHOLD = 0.25

# Fonction de classification des phrases
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

# Fonction pour générer la réponse du chatbot
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for intent in intents['intents']:
                if intent['tag'] == results[0][0]:
                    if 'context_set' in intent:
                        context[userID] = intent['context_set']
                    if not 'context_filter' in intent or (userID in context and 'context_filter' in intent and intent['context_filter'] == context[userID]):
                        return random.choice(intent['responses'])
            results.pop(0)

# Initialisation de Flask
app = Flask(__name__)

context = {}

# Route pour la page d'accueil
@app.route("/")
def home():
    return render_template("index.html")

# Route pour obtenir la réponse du chatbot
@app.route("/get")
def get_bot_response():
    user_input = request.args.get('sentence')
    bot_response = response(user_input)
    return bot_response

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, redirect, url_for, request, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Connexion à la base de données SQLite
def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Créer la base de données et la table des utilisateurs
def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )''')
        conn.commit()

init_db()

# Route d'inscription
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        # Vérifier si l'utilisateur existe déjà
        with get_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if user:
                flash('Email déjà utilisé !', 'danger')
                return redirect(url_for('signup'))

            # Ajouter l'utilisateur à la base de données
            conn.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_password))
            conn.commit()

        flash('Inscription réussie !', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html')

# Route de connexion
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Vérifier l'utilisateur dans la base de données
        with get_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                flash('Connexion réussie !', 'success')
                return redirect(url_for('chatbot'))
            else:
                flash('Email ou mot de passe incorrect', 'danger')
                return redirect(url_for('signin'))

    return render_template('signin.html')

# Route du chatbot (après connexion)
@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('signin'))

    return render_template('chatbot.html')

# Route de déconnexion
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Déconnexion réussie !', 'success')
    return redirect(url_for('signin'))

if __name__ == '__main__':
    app.run(debug=True)

