from gensim.models import Word2Vec

# Expanded and organized training corpus with more cricket terms and general knowledge
sentences = [
    # Royals and Monarchy terms
    ["king", "queen", "prince", "princess", "monarch", "throne", "royal", "kingdom", "empire"],
    ["emperor", "empress", "palace", "crown", "noble", "dynasty", "heir", "rule", "aristocracy"],
    
    # Fruits and food
    ["apple", "banana", "mango", "fruit", "sweet", "juice", "peach", "grape", "berry", "citrus"],
    ["bread", "butter", "jam", "sandwich", "snack", "lunch", "meal", "breakfast", "cereal"],
    ["pizza", "burger", "pasta", "salad", "rice", "spaghetti", "chicken", "beef", "vegetables"],
    
    # Vehicles and transport
    ["car", "truck", "vehicle", "road", "drive", "speed", "highway", "traffic", "garage", "bus"],
    ["bike", "motorcycle", "scooter", "fuel", "engine", "travel", "ride", "journey", "taxi"],
    
    # Programming & Tech
    ["python", "java", "coding", "language", "programming", "debug", "script", "code", "algorithm"],
    ["developer", "engineer", "software", "logic", "compile", "function", "programmer", "tech"],
    ["html", "css", "javascript", "web", "frontend", "backend", "database", "server", "framework"],
    
    # Animals and pets
    ["cat", "dog", "animal", "pet", "cute", "furry", "kitten", "puppy", "bark", "meow"],
    ["zebra", "lion", "elephant", "tiger", "wild", "safari", "zoo", "jungle", "beast", "giraffe"],
    
    # Emotions - Positive
    ["happy", "joyful", "smile", "cheerful", "emotion", "positive", "grateful", "excited", "content"],
    ["laugh", "fun", "peaceful", "calm", "love", "friendly", "hopeful", "relaxed", "serene"],
    
    # Emotions - Negative
    ["sad", "cry", "tears", "pain", "negative", "depressed", "gloomy", "lonely", "fear", "angry"],
    ["upset", "fight", "hate", "rage", "hurt", "suffer", "anxious", "nervous", "stressed"],
    
    # School and education
    ["school", "student", "teacher", "learn", "study", "homework", "class", "subject", "exam"],
    ["book", "pen", "pencil", "notebook", "read", "write", "library", "knowledge", "textbook"],
    
    # Nature
    ["tree", "flower", "grass", "leaf", "forest", "river", "mountain", "earth", "nature", "wildlife"],
    ["sky", "sun", "moon", "rain", "storm", "wind", "cloud", "weather", "season", "climate"],
    
    # Cricket - More detailed terms
    ["cricket", "batsman", "bowler", "wicket", "run", "catch", "ball", "bat", "over", "field"],
    ["six", "four", "boundary", "stumps", "keeper", "umpire", "match", "team", "captain", "innings"],
    ["player", "score", "win", "lose", "toss", "innings", "baller", "spinner", "fast", "allrounder"],
    ["PSL", "IPL", "T20", "ODI", "test", "cricketer", "batting", "bowling", "strike", "century"],
    ["run", "batsman", "bowler", "sixer", "fours", "fielding", "batting", "wicketkeeper", "first-class", "club"],
    
    # Miscellaneous and General terms
    ["technology", "future", "innovation", "development", "research", "science", "discovery", "progress"],
    ["health", "fitness", "exercise", "diet", "workout", "gym", "strength", "cardio", "muscle"],
    ["economy", "business", "trade", "market", "stock", "investment", "finance", "money", "economist"],
    ["art", "painting", "sculpture", "music", "culture", "history", "theater", "literature", "performance"]
]

# Train the model with more epochs, higher vector size, and a larger window
model = Word2Vec(sentences, vector_size=200, window=5, min_count=1, sg=1, epochs=20)

# Save the model
model.save("models/expanded_model_optimized.model")
