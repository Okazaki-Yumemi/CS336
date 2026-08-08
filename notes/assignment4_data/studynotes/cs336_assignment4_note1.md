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
(b) Run your text extraction function on a single WARC file. Compare its output to the extracted text in the corresponding WET file. What differences and/or similarities do you notice? Which extraction seems better