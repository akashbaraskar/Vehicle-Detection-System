from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={'root_dir': 'car damage/test/normal'})

crawler.crawl(
    keyword='normal car',
    max_num=50
)