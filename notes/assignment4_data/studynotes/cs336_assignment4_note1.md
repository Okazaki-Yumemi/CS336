# 2. Filtering Common Crawl

Most researchers don't build a web crawler to source training data for their models. Instead, they use publicly available crawls.

The most popular public webcrawl comes from Common Crawl, a non-profit that provides a free corpus of web pages, making available "over 250B pages spanning 17 years".

However, as we'll see, turning the common crawl (CC) dumps into usable data for language model training takes significant work. For example, the raw data from web pages is in HTML,and we want to extract from it.

In addition,many pages might be of low quality, exact or almost duplicates,have harmful content,or contain sensitive information,and we might want to filter those pages out or remove undesireable parts of their content from our dataset. In this assignment, we'll set up a pipeline that performs several of these steps,turning raw Internet data into a usable training set for language models.

## 2.1 Looking at the data.

Before implementing it, it's always useful to look at the raw data and get a sense of it.

The CC data is made available in three formats:

**WARC**:
  ("Web ARChive format") files contain the raw CC data, which includes page IDs and URLs, metadata and HTTP request details (e.g, date and time of the request, server IP address), and of course the raw content of the page

**WAT**:
  ("Web Archive Transformation") files contain higher-level metadata,extracted from WARC files and dumped as a JSON object. For example, for HTML pages, this includes a list of links from that page and the page title.

**WET**:
  ("Web Extracted Text") files contain extracted plain text from the raw HTML pages.



**Problem: Looking at Common Crawl**
(a): Use the copy of the WARC file provided at the path above. Let's look at the first page in this file.

example:
```
WARC/1.0
WARC-Type: metadata
WARC-Date: 2026-03-05T08:39:38Z
WARC-Record-ID: <urn:uuid:bf49ed92-a2dd-4c08-8ee2-8e47e32f327f>
Content-Length: 201
Content-Type: application/warc-fields
WARC-Warcinfo-ID: <urn:uuid:8037c9e0-d61b-4def-9f1b-6aac414cb992>
WARC-Concurrent-To: <urn:uuid:410fc213-8e9f-4e18-9416-66b4bac5cdce>
WARC-Target-URI: http://020bld.cn.vauofnj.cn/html/991f998999.html

fetchTimeMs: 554
charset-detected: UTF-8
languages-cld2: {"reliable":true,"text-bytes":3904,"languages":[{"code":"zh","code-iso-639-3":"zho","text-covered":0.99,"score":1869.0,"name":"Chinese"}]}
```


(b): Note that the WET files contain HTTP headers that are not part of the extracted text contents. If you look at the first example, you will see that it contains text that was extracted from the raw HTML you just saw.

Notice that much of the extracted text is reminiscent of the HTML structure, and not actually the page's main content. Are there parts of the text you see that you think should have been filtered out by the extractor? Think about the quality of this text as training data: what might go wrong in training a model on the text that looks like this? Conversely, what useful information can a model potentially extract from this page?

```
WARC:
metadata
HTTP headers
大量 HTML tags
网页正文
导航/脚本/模板等

          ↓ extractor

WET:
metadata
抽出来的可见文本
但仍可能混着：
导航栏
菜单
footer
cookie notice
重复模板
无意义短文本
```

数据很脏
有赌博网站
然后还有黄色网站mlgb的

(c) what makes a good training example is highly contextual. Describe an application domain for which this example might be useful to have in the training data, and one where it might not be.

> 例如我爬出来一些技术blog,这些地方就是好的训练数据，有完整可读的连续英文。 有些爬出来是黄色网站，这些就是不好的训练数据，可能会导致模型生成不良内容。

(d)  Let’s look at some more examples to get a better sense of what’s in the Common Crawl. Look through 25 more WET records. For each record, very briefly comment on the document’s language (if you can identify it), the domain name, what type of page it is, etc. How many examples does it take until you see what you’d deem a “high-quality” webpage?


## 2.2 HTML to text conversion

As you might have realized from looking at the WARC and WET files above, extracting text from HTML is challenging.

Typically, any extraction procedure will look for visible content in the HTML(such as <p> tags,which are supposed to contain blocks of the text). But that can still extract much more than what we would perceive as the main content of a page when opening it in a Web browser. For example, when opening stack StackOverflow,the main content is in the questions and answers, but technically the menu options, links to unrelated pages in other StackExchange, footer , links to sign up or log in —— those are all visible text, and it is challenging to distinguish those reliably from the page's main content.

Many tools implement text extraction pipelines. In this assignment, we will use the Resiliparse library for performing text extraction.
Resiliparse will also help with an even more basic problem: detecting the text encoding of the bytes containing the raw content. Although most pages on the Web are encoded in UTF-8, our text extraction pipeline should be robust to other encodings as well.

**Problem: HTML to text conversion**
(a) Write a function that extracts text from a byte string containing raw HTML. Use resiliparse.extract.html2text.extract_plain_text to perform the extraction. This function  needs a string, so you will need to first decode the byte string into a Unicode string. Be aware that the input byte string might not be encoded in UTF-8, so your function should be able to detect the encoding in case UTF-8 fails. Resiliparse also offers resiliparse.parse.encoding.detect_encoding(), which might be useful

> 见codenote1

(b) Run your text extraction function on a single WARC file. Compare its output to the extracted text in the corresponding WET file. What differences and/or similarities do you notice? Which extraction seems better

>提取出来的是纯文本，非常好

## 2.3 Language identification

The web contains pages written in thousands of languages. But training a multilingual model that can effectively make use of such diverse data at scale is challenging at most compute budgets. Thus, many language modeling training sets derived from Common Crawl contain data from a limited set of languages.

A useful library for this purpose is fastText. which provides efficient text classifiers. The library provides both the infrastructure to train classifiers on your own data and a collection of pre-trained models, including for language identification. You can download the fastText language identification model. The model is available at /shared-data/classidiers/lid.176.bin

Typically, language filters use a score given by classifier to decide whether to keep a given page. Use the fastText language identification classifier to implement a language identification filter , which should give a non-negative score for how confident it is in the prediction.

**Problem: Language identification**
(a) Write a function that will take a Unicode string and identify the main language that is present in this string. Your function should return a pair, containing an identifier of the language and a score between 0 and 1 representing its confidence in that prediction.

> 见codenote2


(b)  The behavior of language models at inference time largely depends on the data they were trained on. As a result, issues in the data filtering pipeline can result in problems downstream. What issues do you think could arise from problems in the language identification procedure? In a higher-stakes scenario (such as when deploying a user-facing product), how would you go about mitigating these issues?

> 多保留一点有噪声的数据，和错误删除大量高质量英文数据，哪一个对最终 LM 更糟？ thershold

(c) Run your language identification system on text extracted from the WARC files (via your previously-implemented text extraction function). Manually identify the language in 20 random examples and compare your labels with the classifier predictions. Report any classifier errors. What fraction of documents are English? Based on your observations, what would be a suitable classifier confidence threshold to use in filtering?

```bash
(cs336-data) (base) soyo@localhost:~/projects/CS336-2026/assignments/a4-data$ uv run cs336_data/languageIdentification.py 
Language: zh, Confidence: 0.9608736634254456
Language: zh, Confidence: 0.9902801513671875
Language: zh, Confidence: 0.9242303967475891
Language: zh, Confidence: 0.9970689415931702
Language: zh, Confidence: 0.9923343062400818
Language: zh, Confidence: 0.9676010608673096
Language: zh, Confidence: 0.9651224613189697
Language: en, Confidence: 0.7118707299232483
Language: ru, Confidence: 0.994608461856842
Language: ru, Confidence: 0.9790343046188354
Language: de, Confidence: 0.9181560277938843
Language: zh, Confidence: 0.993803858757019
Language: zh, Confidence: 0.8872329592704773
Language: el, Confidence: 0.9986318349838257
Language: en, Confidence: 0.8817716836929321
Language: zh, Confidence: 0.8872559070587158
Language: zh, Confidence: 0.9762156009674072
Language: zh, Confidence: 0.9676953554153442
Language: en, Confidence: 0.11044929921627045
Language: en, Confidence: 0.9568921327590942
Language: en, Confidence: 0.11044929921627045
Language: ja, Confidence: 0.9922927021980286
Language: zh, Confidence: 0.9486950635910034
Language: zh, Confidence: 0.9809586405754089
Language: nl, Confidence: 0.8178791403770447
Language: zh, Confidence: 0.9890671372413635
Language: zh, Confidence: 0.9070631265640259
Language: zh, Confidence: 0.9882130026817322
Language: zh, Confidence: 0.9910210967063904
Language: zh, Confidence: 0.6539320349693298
Language: zh, Confidence: 0.5781425833702087
Language: ru, Confidence: 0.9842963814735413
Language: ru, Confidence: 0.9899467825889587
Language: en, Confidence: 0.11044929921627045
Language: zh, Confidence: 0.9861137270927429
Language: zh, Confidence: 0.8912011981010437
Language: zh, Confidence: 0.9541097283363342
Language: zh, Confidence: 0.985442578792572
Language: zh, Confidence: 0.9876883625984192
Language: zh, Confidence: 0.974861741065979
Language: zh, Confidence: 0.9778426289558411
Language: ru, Confidence: 0.9729216694831848
Language: zh, Confidence: 0.9601818323135376
Language: ru, Confidence: 0.9998847246170044
Language: en, Confidence: 0.9218418598175049
Language: de, Confidence: 0.37443867325782776
Language: ko, Confidence: 0.9302152991294861
Language: en, Confidence: 0.8041372299194336
Language: en, Confidence: 0.11044929921627045
Language: en, Confidence: 0.926743745803833
Language: tr, Confidence: 0.9926760196685791
Language: en, Confidence: 0.11044929921627045
Language: zh, Confidence: 0.9960695505142212
Language: en, Confidence: 0.36531123518943787
Language: en, Confidence: 0.9453595876693726
Language: en, Confidence: 0.11044929921627045
Language: id, Confidence: 0.8264915943145752
Language: zh, Confidence: 0.9797539114952087
Language: zh, Confidence: 0.9934940934181213
Language: fr, Confidence: 0.9906957149505615
```



## 2.4 Personally identifiable information.

The web contains large quantities of information that can be used to reach or identify individuals. Such as email addresses . phone numbers, IP address.

We might not want a user-facing language model to output such information about real people, and a common step is to mask out these pieces of information in the training dataset.

**Problem: Personally identifiable information**:

(a) Write a function to mask out emails. Your function will take a string as input, and replace all instances of email addresses with the string "|||EMAIL_ADDRESS|||". To detect email addresses, you can look up regular expressions that do this reliably.
> 见codenote3
(b) Write a function to mask out phone numbers. Your function will take a string as input, and replace all instances of phone numbers with the string "|||PHONE_NUMBER|||". Doing this reliably can be extremely challenging, as phone numbers might be written in an extremely diverse set of formats, but you should try to capture at least the most common phone number formats used in the United States, and be robust to minor syntactic deviations.
> 见codenote3
(c) Write a function to mask out IP addresses. For this problem, it is enough to focus on IPv4 addresses (4 numbers up to 255 separated by points). Your function will take a string as input, and replace all instances of IP addresses with the string "|||IP_ADDRESS|||".
> 见codenote3
(d) What problems do you think might arise downstream in a language model when these filters are naïvely applied on the training set? How might you mitigate these issues?
(e)  Run your PII masking functions on text extracted from the WARC files (via your previouslyimplemented text extraction function). Look through 20 random examples where a replacement was made; give some examples of false positives and false negatives.