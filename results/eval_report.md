# Evaluation: our model vs GPT-2

## Held-out FineWeb-Edu text

1,036 documents (5.0 MB) of FineWeb-Edu `sample-10BT`, starting after document 7,500,000 (all training data comes from earlier documents).

| Model | Bits/byte (lower is better) | Loss/token | Perplexity | Bytes/token | Tokens |
|---|---|---|---|---|---|
| Ours (110M) | 1.0004 | 3.2032 | 24.61 | 4.62 | 1,083,383 |
| GPT-2 small (124M) | 1.0093 | 3.2358 | 25.43 | 4.63 | 1,081,994 |
| GPT-2 medium (355M) | 0.9262 | 2.9694 | 19.48 | 4.63 | 1,081,994 |
| GPT-2 large (774M) | 0.8801 | 2.8215 | 16.80 | 4.63 | 1,081,994 |
| Ours (336M) | 0.8966 | 2.8708 | 17.65 | 4.62 | 1,083,383 |

Loss per token and perplexity depend on each model's tokenizer; compare models by bits per byte.

## HellaSwag (validation)

| Model | acc | acc_norm | Examples |
|---|---|---|---|
| Ours (110M) | 27.71% | 27.25% | 10042 |
| GPT-2 small (124M) | 28.67% | 29.52% | 10042 |
| GPT-2 medium (355M) | 32.20% | 37.67% | 10042 |
| GPT-2 large (774M) | 34.92% | 43.12% | 10042 |
| Ours (336M) | 30.07% | 32.87% | 10042 |

Random guessing scores 25%.

## Generation speed (batch 1, tokens/s)

| Model | With KV cache | Without cache | Speedup |
|---|---|---|---|
| Ours (110M) | 98.4 | 97.2 | 1.0x |
| GPT-2 small (124M) | 141.9 | 117.5 | 1.2x |
| GPT-2 medium (355M) | 77.1 | 62.5 | 1.2x |
| GPT-2 large (774M) | 53.6 | 37.5 | 1.4x |
| Ours (336M) | 49.7 | 50.0 | 1.0x |

## Generations

### factual: "The capital of France is"

**greedy**

- *Ours (110M):* ' the capital of the French Republic. The capital of France is the capital of the French Republic. The capital of France is the capital of the French Republic. The capital of France is the capital of the French Republic. The capital of France is the capital of the French Republic. The capital of France is'

- *GPT-2 small (124M):* ' a city of the French, and the capital of the French is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.'

- *GPT-2 medium (355M):* ' a city of the future.\n\nThe city of Paris is a city of the future.\n\nThe city of Paris is a city of the future.\n\nThe city of Paris is a city of the future.\n\nThe city of Paris is a city of the future.\n\n'

- *GPT-2 large (774M):* ' a city of contrasts. The city is a vibrant, cosmopolitan metropolis, with a vibrant, cosmopolitan culture. The city is a vibrant, cosmopolitan metropolis, with a vibrant, cosmopolitan culture.\n\nThe city is a vibrant, cosmopolitan metropolis, with a vibrant,'

- *Ours (336M):* ' Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\nThe'

**temperature 0.8, top-k 50**

- *Ours (110M):* ' Duchy of Durer and is located in the centre of the Netherlands. The capital of the Netherlands is Duchy of Durer. The city of Durer is famous for its historical monuments and the famous castle of Durer. The castle of Durer was built'

- *GPT-2 small (124M):* " home to dozens of monuments, especially the Louvre. But as the Louvre is the most iconic of the country's museums, it is increasingly under threat.\n\nThe Louvre has been the home of hundreds of thousands of visitors and is one of the nation's most important art collections for centuries"

- *GPT-2 medium (355M):* ' home to dozens of monuments, some of them nearly one thousand years old, but it is also an unusual mecca for the French fashion scene, where streetwear brands, from Mango to Tom Ford, are known to arrive at the latest shows with bags stuffed full of designer labels. Many of these'

- *GPT-2 large (774M):* " home to dozens of monuments, some of which were built thousands of years ago, but many of the remaining structures were not built until the last century. Here's a look at some of the most famous landmarks in the city.\n\n1. Notre Dame Cathedral\n\nBuilt in 1499 for the"

- *Ours (336M):* ' Paris.\nPeople who are not very religious are called Protestants or Catholics. Catholics are people that believe in one God, but not in Jesus Christ. Catholics are people who believe in one God, and not in Jesus Christ.\nThe most important people in France are the nobles. Their main duty is'

**temperature 0.8, top-p 0.9**

- *Ours (110M):* ' Léopoldville. Léopoldville, also known as Léopoldville, is a capital city and industrial complex in the south of France. Léopoldville is the location for the city of Léopoldville (the city of Léopoldville). Léopoldville is'

- *GPT-2 small (124M):* " being targeted for a new anti-Islam video, because the country's leader said there was nothing to stop Muslims from preaching their religion.\n\nThe leader of the Flemish National Party, Jean-Pierre Niziol, said on Tuesday that the video, featuring women who were dressed as cow"

- *GPT-2 medium (355M):* " the city of Paris. It's the most populous city in France and the capital of the French Republic. It's also the world's most populous city and the largest city by population. Paris is also the world's most polluted city.\n\nThe French city is home to many cultural and historical landmarks"

- *GPT-2 large (774M):* ' the city of Paris. It is the most populous city in France and the capital of the French Republic. It is located on the banks of the Seine River, at the confluence of the Rhone and the Seine, in the heart of the city.\n\nThe city is built around'

- *Ours (336M):* ' Paris. Paris is the cultural capital of France, and it is the capital of the United Kingdom. It is also the largest city in the European Union. Paris is the main city of France, and it is the second most important city in the world. It is also the largest city in France,'

### factual: "Water is made of two elements:"

**greedy**

- *Ours (110M):* ' water and air. The first element is water, which is made of water and air. The second element is water, which is made of water and air. The water is made of air and the air is made of air. The water is made of air and the air is made of air.'

- *GPT-2 small (124M):* ' water and oxygen. Water is the most abundant element in the universe, and oxygen is the most abundant element in the universe.\n\nWater is the most abundant element in the universe, and oxygen is the most abundant element in the universe. Water is the most abundant element in the universe, and oxygen'

- *GPT-2 medium (355M):* ' water and oxygen. The water molecules in water are called water molecules. Oxygen molecules are made of carbon atoms. The oxygen atoms in water are called oxygen atoms. The water molecules in water are called water molecules.\n\nThe water molecules in water are called water molecules. Oxygen molecules are made'

- *GPT-2 large (774M):* ' water and oxygen. Oxygen is the most abundant element in the universe, and it is the most abundant element in the universe. Water is made of two elements: hydrogen and oxygen. Hydrogen is the most abundant element in the universe, and it is the most abundant element in the universe.\n'

- *Ours (336M):* '\n- Water is a liquid that is made up of two different elements:\n- Water is a gas that is made up of two different elements:\n- Water is a liquid that is made up of two different elements:\n- Water is a gas that is made up of two different elements:'

**temperature 0.8, top-k 50**

- *Ours (110M):* ' chlorine and oxygen. For instance, the chlorine atom is one of the elements of the element that makes up the element, and the oxygen atom is one element of the element oxygen. The other element that makes up the element’s chlorine atom is oxygen. The chlorine atom is the “fountiful'

- *GPT-2 small (124M):* ' sodium and calcium, and both are important for the body to function properly. Sodium is made of an organic molecule called sodium hydroxide, or sodium hydroxide, called sodium carbonate. This is why it is called the salt of the earth, which makes water. Ca 2+ is also'

- *GPT-2 medium (355M):* " hydrogen and oxygen, and since the presence of an excess of one of these elements means the presence of an excess of water in the mix, everything on Earth is made up of water. That is, it's a pretty self-evident fact that water exists somewhere.\n\nAnd yet for some"

- *GPT-2 large (774M):* ' hydrogen and oxygen, and since the presence of the hydrogen atom in water can cause it to boil, the amount of hydrogen in the water is important. Scientists can measure the concentration of hydrogen in the water by using a spectrometer. The spectrometer measures the absorption of light by the water molecule'

- *Ours (336M):* '\n- Calcium hydroxide, which is a mineral that makes the water soft and malleable.\n- Magnesium hydroxide\nMagnesium hydroxide is a strong chemical that dissolves and dissolves. It’s not made of pure minerals.\nMagnesium hydroxide is used'

**temperature 0.8, top-p 0.9**

- *Ours (110M):* ' water and oxygen.\nAs you can see from the chart, the air contains about one-third of the oxygen in the air.\nThis means that you can breathe in water and use it for many other purposes, from food to medicine.\nThe amount of oxygen in the air depends on how'

- *GPT-2 small (124M):* ' the organic molecule known as triphosphate and the protoplasmic triphosphate found in water. Triphosphate and protoplasmic triphosphate are both more abundant than organic material. Because water is an energy source, it is easy for water molecules to get into'

- *GPT-2 medium (355M):* ' hydrogen and oxygen. The hydrogen bonds in water are called protons and the oxygen bonds are called electrons. Water is also a solid. Although it is made of water, it is not a liquid. A liquid is a solid, but it cannot be covered by water. It is not necessary to be'

- *GPT-2 large (774M):* " hydrogen and oxygen. The second element is most important because it's the element that makes water boil and make life possible on Earth. But the first element, hydrogen, is also the element that gives you life.\n\nHydrogen is an incredibly common element found in nature, as well as being used"

- *Ours (336M):* ' water and oxygen.\nThere are eight elements (A, B, C, D, E, F, G, and H) in the universe. They are called the elements because they are the fundamental units of all things. Each element has a unique shape and can be easily identified by its own'

### explanatory: "Photosynthesis is the process by which"

**greedy**

- *Ours (110M):* ' plants convert light energy into chemical energy. The process of photosynthesis is the process of photosynthesis.\nPhotosynthesis is the process by which plants convert light energy into chemical energy.\nPhotosynthesis is the process by which plants convert light energy into chemical energy.\nPhotosynthesis is the process by which plants convert'

- *GPT-2 small (124M):* ' the air molecules in the air are converted into sugars. The process is called the "sugar cycle."\n\nSugar is the most abundant form of food in the world. It is the most abundant food in the world because it is the only food that is made from the same chemical reactions that'

- *GPT-2 medium (355M):* ' plants use sunlight to produce energy. The process is a complex process that involves the production of carbon dioxide, water, and oxygen.\n\nThe process of photosynthesis is a complex process that involves the production of carbon dioxide, water, and oxygen.\n\nThe process of photosynthesis is a complex'

- *GPT-2 large (774M):* ' plants convert sunlight into energy. It is the process that gives plants their green color.\n\nThe process is called photosynthesis.\n\nThe process is called photosynthesis.\n\nThe process is called photosynthesis.\n\nThe process is called photosynthesis.\n\nThe process is called photos'

- *Ours (336M):* ' plants convert sunlight into chemical energy. The process is essential for the growth of plants, and is also essential for the survival of many animals.\nPhotosynthesis is a process that occurs in plants and animals. It is a process that takes place in the cells of plants and animals. The process is essential'

**temperature 0.8, top-k 50**

- *Ours (110M):* ' sunlight changes from light-colored to dark brown to light-colored, with different colors being applied.\n|Sequence||Color||Brightness||Diffusion|\n|Brightness||Yellow||Closest||X-Y|\n|Brightness||Yellow'

- *GPT-2 small (124M):* ' organic chemicals are absorbed by plants and soil. It is the process of photosynthesis that is most familiar to me. This article will show you how it happens, what it can be, and how it is used.\n\n\nOrganic photosynthesis is one of the basic processes being used by plants to'

- *GPT-2 medium (355M):* ' organic matter (food) is converted into energy. Plants are also photosynthetic organisms.\n\nPlant cells utilize oxygen, water, and carbon dioxide as organic matter called CO2 for photosynthesis. Plants are also self-sustaining, utilizing sunlight for photosynthesis and oxygen for respiration'

- *GPT-2 large (774M):* ' plants and animals convert sunlight into food. The photosynthetic processes of photosynthetic organisms are the same as those found in other organisms. Photosynthetic organisms produce oxygen from CO2, the primary byproduct of photosynthesis. These photosynthetic organisms also produce carbon dioxide and water.<|endoftext|>'

- *Ours (336M):* ' plants use light energy to convert carbon dioxide into sugar.\nPhotosynthesis is a complex process that involves a number of different processes.\nPhotosynthesis occurs in a plant:\n- The process involves photosynthesis in which the plant’s own photosynthesis occurs.\n- The process involves the interaction between two separate'

**temperature 0.8, top-p 0.9**

- *Ours (110M):* ' plants convert sunlight to energy for photosynthesis. Since photosynthesis is not a natural process, it is usually not done at all, because the energy it produces is not available at the cellular level. Thus, it is important to consider the bioenergy efficiency of plants in terms of their relative efficiency.\nThe biological'

- *GPT-2 small (124M):* ' the carbon dioxide that is released into the atmosphere is converted to heat in the form of heat energy. The heat energy from sunlight and the heat energy from carbon dioxide are absorbed by the cells of the plant, which then produce energy, in turn. Plants absorb more heat energy from the sun than they can'

- *GPT-2 medium (355M):* ' plants use sunlight to produce energy. Photochemical reactions occur in the leaf membrane, between the leaves and the underside of the leaves. The process is an efficient way of converting sunlight into electricity.\n\nMost photosynthesis occurs on the leaves of plants, but some species are capable of producing electricity through their'

- *GPT-2 large (774M):* ' plants and animals convert sunlight into sugars and other nutrients. With the help of sunlight, plants produce their own food and other animals eat plants and their food. Plants are also able to generate oxygen. Plants use photosynthesis to help them grow, reproduce, and reproduce more.\n\nThe photosynthetic'

- *Ours (336M):* " plants convert sunlight to chemical energy for growth and cell division. This process is essential for growth and development of all plants, from the smallest to the largest. Photosynthesis occurs in the cell's mitochondria, which are the powerhouses of the cell. The energy from sunlight is used to synthesize proteins, sugars"

### explanatory: "The seasons change because"

**greedy**

- *Ours (110M):* ' of the seasons. The seasons change because of the seasons. The seasons change because of the seasons. The seasons change because of the seasons. The seasons change because of the seasons. The seasons change because of the seasons. The seasons change because of the seasons. The seasons change because of the seasons.'

- *GPT-2 small (124M):* ' of the weather.\n\nThe season is over.\n\nThe season is over.\n\nThe season is over.\n\nThe season is over.\n\nThe season is over.\n\nThe season is over.\n\nThe season is over.\n\nThe season is over.'

- *GPT-2 medium (355M):* ' of the seasons.\n\nThe seasons change because of the seasons.\n\nThe seasons change because of the seasons.\n\nThe seasons change because of the seasons.\n\nThe seasons change because of the seasons.\n\nThe seasons change because of the seasons.\n\nThe seasons change because'

- *GPT-2 large (774M):* ' of the sun.\n\nThe sun is the source of light, and the seasons change because of the sun.\n\nThe seasons change because of the sun.\n\nThe seasons change because of the sun.\n\nThe seasons change because of the sun.\n\nThe seasons change because of'

- *Ours (336M):* ' of the earth’s rotation. The seasons are caused by the tilt of the earth’s axis. The tilt of the earth’s axis is caused by the tilt of the earth’s axis. The tilt of the earth’s axis is caused by the tilt of the earth’s axis'

**temperature 0.8, top-k 50**

- *Ours (110M):* ' of changes in seasons and the changing climate.\nThe seasons change so much that the seasons that are expected are not so often.\nThe season changes because of changes in weather conditions. In this case, the seasons change because the seasons change so that the seasons change because the weather changes because of changes'

- *GPT-2 small (124M):* ' of the seasons, and so do the seasons. For example, the season 1 season is the season that starts on April 1. The season 2 season is the season that begins on April 1.\n\nThere are no seasons in Season 1.\n\nSeason 1 seasons\n\nSeason 2 seasons\n'

- *GPT-2 medium (355M):* " of the seasons, and since my last post I've been having to make a very hard decision about what I wanted to post this week. After thinking about it for a while and re-reading the posts I've already written and taking some time to think about the subject matter, I've decided to"

- *GPT-2 large (774M):* ' of the sun, and since the earth is the center of the solar system, we can make an educated guess that the seasons change in accordance with the change in the sun. The reason why the sun changes direction is because the earth spins around its axis, which makes the seasons change because the seasons are'

- *Ours (336M):* ' of changes in the level of light and/or the angle of the sun. The Sun also changes with time, so that the length of days and hours are changed by the Sun. In this way, the night time is the same length as the day. The seasons are different because there are different'

**temperature 0.8, top-p 0.9**

- *Ours (110M):* ' of a new school year, but the children are also in school in a different way.\nChildren are learning about the seasons in different ways.\nTogether, they’re learning about the weather, which is a natural part of life, and about the seasons.\nThis week, we’'

- *GPT-2 small (124M):* ' the band creates new music and artwork in their studio.\n\n"We\'re doing an album together that we have been working on for about a year and a half, so we want to do it as soon as possible," said Chris Brown. "I think I\'ll take a couple weeks off,'

- *GPT-2 medium (355M):* " the sun rises and sets, which is why it's so important to follow the sun. You can follow the sun by using solar tracking and using an altimeter to measure your position. You can also follow the sun by using a solar compass.\n\nIf you are in the United States or Canada"

- *GPT-2 large (774M):* ' the earth moves. The world is not always the same. At the end of each season, the ice caps and the continents melt and the sea levels rise. But for many parts of the world, the change is slow and gradual. And for some people, they are worse off.\n\nThe'

- *Ours (336M):* ' of the changes in light and temperature.\nThe average daily maximum temperature is 38.0 degrees Fahrenheit.\nThe average daily minimum temperature is 26.0 degrees Fahrenheit.\nThe average daily maximum temperature is 36.0 degrees Fahrenheit.\nThe average daily minimum temperature is 32.0 degrees Fahrenheit.'

### creative: "Once upon a time, in a small village by the sea,"

**greedy**

- *Ours (110M):* ' a man was a man of the sea, and the sea was a man of the sea. The sea was a man of the sea, and the sea was a man of the sea. The sea was a man of the sea, and the sea was a man of the sea. The sea was'

- *GPT-2 small (124M):* " a man named T'Challa was a young man who had been born in the city of T'Challa. He was a man of great wealth and power, and had a great reputation for his prowess. He was a man of great strength, and was a man of great courage. He"

- *GPT-2 medium (355M):* ' a young man named Tatsuya was a very good student. He was a very good student, and he was also a very good man. He was a very good man, and he was also a very good man.\n\nTatsuya was a very good man, and he was also'

- *GPT-2 large (774M):* ' there lived a man named J.R.R. Tolkien. He was a man of many talents, and he was also a man of many passions. He was a man of many interests, and he was also a man of many passions. He was a man of many interests, and he was'

- *Ours (336M):* ' a young man named John was walking along the beach. He was looking for a place to stay. He was looking for a place to live. He was looking for a place to live. He was looking for a place to live.\nJohn was looking for a place to live. He was looking'

**temperature 0.8, top-k 50**

- *Ours (110M):* ' there could be no more than a few men to die. In the beginning, the whole system was a very small one. The villagers were very busy. They had to carry the house, put up the clothes. A little book from the library of the school, which the young man had read,'

- *GPT-2 small (124M):* " a man named Rohan had gathered his family. For some reason, he wanted to go to an inn or get some beer. His family were all married, and he had some friends out there.\n\nThe innkeeper's brother, Voldie, had died. No one knew who he"

- *GPT-2 medium (355M):* ' a young man, named Peter, lived a quiet life. He owned a shop called the Cafe de San Joaquin, where he would prepare everything from pasta to cheeses from fresh produce. In the summer, his customers would come to visit. One day, the owner asked Peter to bring him some'

- *GPT-2 large (774M):* ' by the sea, in a small town called the village, by the sea, we lived in an unusual and beautiful world.\n\nWhen I was six, I had the best job of my life as a part time teacher at school and I was able to get away from being a part time teacher'

- *Ours (336M):* ' there lived a woman who had a very important job.\nShe had a large family and was a very important member of the village\nBeside that, in a village by the sea, there lived a woman who worked the same job as she did. She had a very important job.\n'

**temperature 0.8, top-p 0.9**

- *Ours (110M):* " there were a handful of women all in one small boat. Many in a small village, at the village's centre, stood in the water. She was the only person who could afford to support herself. She was a small woman in one of the smaller boats. She had a stone wall of her"

- *GPT-2 small (124M):* " the inhabitants of a land of six hundred and twenty-five miles in circumference, stood on a little island. The island was in the midst of an immense ocean of dry, sandy land. There were so many trees there, that it was difficult to imagine a man's existence without trees.\n\n"

- *GPT-2 medium (355M):* ' a band of brothers and sisters grew up, and they were, in every sense of the word, the purest of all. They were all peasants and hardy, hardworking, strong-minded, honorable, devoted to their home, their family, and their land. They were religious, free'

- *GPT-2 large (774M):* ' there lived a young woman who loved her dogs and the ocean, and she loved nothing more than to go to the beach and play in the waves.\n\nOne day, the young woman went to the beach to play with her dogs, and, when she returned, she found that they had grown'

- *Ours (336M):* ' there lived a king called Si-il-yin. He was a very handsome, handsome man. He had three wives, and he gave them three sons, and they became very rich. When the king was growing old, he came to the shores of the lake, and a large canoe came'

### completion: "The most important thing about learning is"

**greedy**

- *Ours (110M):* ' that it is not a one-time event. It is a time of discovery and discovery. It is a time of discovery and discovery.\nThe most important thing about learning is that it is not a one-time event. It is a time of discovery and discovery.\nLearning is a process'

- *GPT-2 small (124M):* ' that you can learn it.\n\nThe most important thing about learning is that you can learn it.\n\nThe most important thing about learning is that you can learn it.\n\nThe most important thing about learning is that you can learn it.\n\nThe most important thing about learning is'

- *GPT-2 medium (355M):* " that it's not about the material. It's about the process.\n\nI've been doing this for a long time. I've been doing it for a long time, and I've been doing it for a long time.\n\nI've been doing it for a long time, and"

- *GPT-2 large (774M):* ' that you learn.\n\nThe most important thing about learning is that you learn.\n\nThe most important thing about learning is that you learn.\n\nThe most important thing about learning is that you learn.\n\nThe most important thing about learning is that you learn.\n\nThe most'

- *Ours (336M):* ' that it is fun.\nLearning is fun because it is fun.\nLearning is fun because it is fun.\nLearning is fun because it is fun.\nLearning is fun because it is fun.\nLearning is fun because it is fun.\nLearning is fun because it is fun.\n'

**temperature 0.8, top-k 50**

- *Ours (110M):* ' that you don’t get the answer right. You can’t learn as much as that. But you should know that it’s really important that you learn before you learn. In this article I’m going to talk about the importance of learning and learning. So that you can have confidence'

- *GPT-2 small (124M):* " getting it right, and that means getting it right with the right mindset. This is why we don't want to say that this book is everything you need to know, but rather, we want to make it a lot more fun. Take this book, for example.\n\nA couple of months"

- *GPT-2 medium (355M):* " to not get discouraged. You can learn all the time, but you can also learn to make mistakes.\n\nLearn from other mistakes. After all, you have a lot to learn about yourself.\n\nOne of the biggest mistakes is to give too much thought to what you're asking for."

- *GPT-2 large (774M):* ' getting the right information. You should start with the basics, then develop the knowledge along the way. And when you have really good information, you can start moving forward, adding to it, adding to it, adding to it.\n\nBut that is not the only way to learn. You can'

- *Ours (336M):* ' not so much the level of understanding achieved nor the amount of practice, but rather the quality that you have actually achieved.\nThe key to learning how to speak correctly is to be comfortable with the grammatical structure of the language you are learning.\nIf you are learning a language that you cannot speak,'

**temperature 0.8, top-p 0.9**

- *Ours (110M):* ' that it provides a huge number of opportunities for students to become self-directed and self-motivated. It also provides a wide range of opportunities for students to be active, creative and self-directed.\nWe all know that having a well-rounded educational system allows students to grow. It'

- *GPT-2 small (124M):* ' the application of your experience. How often will you learn from someone who has not done so? How often will you learn from someone who has done so and have done so many things wrong? How often will you learn from someone who has done so, yet is still learning from them?\n\nThe'

- *GPT-2 medium (355M):* ' being able to learn. What we learn, and what we teach others to learn, is important. To be able to learn and pass on our knowledge and ideas is important, too.\n\nBy learning from the perspectives of other teachers, you can get a more solid foundation for your own teaching.'

- *GPT-2 large (774M):* " being willing to learn. If you're not willing to learn, you're not going to be able to make good decisions. You're going to be stuck with the same mistakes over and over. That's the worst thing you can do.\n\nI was in a situation recently where I was really"

- *Ours (336M):* ' not being able to see the problem in front of you. It is a difficult task, but not impossible.\nAnd while the first part of this series will be on identifying the problem, the next part will be on understanding how to solve the problem. It is a long process and you will have'
