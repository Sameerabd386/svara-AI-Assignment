# SvaraAI Reply Classifier - Short Answer Reasoning (Part C)

## Question 1: If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

Honestly, with just 200 samples I'd probably start with data augmentation techniques. I could use things like back-translation where I translate the text to another language and back to English to get slightly different versions of the same meaning. I might also try paraphrasing tools or simple word substitutions with synonyms to create more training examples. 

Transfer learning would be my main approach though - using pre-trained models like DistilBERT that already understand language from huge datasets, so I'm not starting from scratch. I'd also look into active learning where I strategically pick which new samples to label based on what the model is most uncertain about, so I get the most bang for my buck with each new label.

## Question 2: How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?

I'd definitely need to test the model on different types of emails and writing styles to see if it's biased toward certain groups or communication patterns. For example, does it incorrectly classify professional vs casual writing styles, or does it have trouble with emails from different industries?

I'd set up some kind of monitoring system to track the predictions over time and flag when something looks off - like if suddenly 90% of predictions are "negative" which might indicate something went wrong. Having humans review a sample of predictions regularly would be important too, especially the ones where the model isn't very confident. I'd also want a feedback system where sales reps can report when the classifier got something wrong, so I can keep improving it.

## Question 3: Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I'd definitely include specific context about the company I'm reaching out to in the prompt - things like their recent news, what industry they're in, mutual connections, or specific pain points their industry faces. The more specific information I can feed the model, the better.

I'd probably use few-shot prompting where I show the model examples of good personalized openers versus generic ones, so it learns the difference. I'd also be explicit in my instructions to avoid generic phrases like "I hope this email finds you well" and instead focus on mentioning at least 2-3 specific things about their company or situation. Maybe I'd even add constraints like "mention something from their recent LinkedIn post" or "reference their company's recent expansion" to force the personalization.
