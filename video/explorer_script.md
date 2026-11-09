# Video: what the explorer is

For Shamekh to read out loud. About two minutes. Text in [brackets] is a stage direction.

[Show the explorer window.]

This is the prediction explorer. It shows how CLIP, a model from OpenAI, guesses whether a photo is from Austria or from Australia. CLIP was trained on photos and text from the internet, and we don't train it for this task. We give it the photo and a short sentence for each country, like "a photo I took in Austria", and it picks the sentence that matches the photo better.

We used 200 photos from an open dataset, 100 from each country. For every photo, the explorer shows three bar charts: one with English names, one with German names, and one with a mix. The bars show how sure CLIP is. Red is Austria, blue is Australia. The title says whether it was right. At the bottom are its top five guesses out of all 211 countries.

[Click Next a few times.]

The main finding is about the names. With English names, CLIP got 85 of the 100 Austria photos right. With German names it got about 60. CLIP was built for English, so German is a harder test for it. For one photo, English gives 88 percent Austria, and German gives 6 percent.

[Go to photo 4, Austria_003.]

Another finding is a strange guess. For 24 of the photos, CLIP's top guess was "Reunion", mostly photos of groups of people. "Reunion" is also an English word for a meeting. When we renamed the country to "Réunion", that guess disappeared. So CLIP seems to match the English word, not the island.

[Tick "Only mistakes".]

In total, 47 of the 200 photos are wrong in at least one of the three ways. We checked 300 more photos from the training part of the dataset, and the pattern holds there too.
