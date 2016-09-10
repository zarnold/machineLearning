# This script get and make frenchtext dataset from news corpus

# This function will convert text to lowercase and remove special characters
normalize_text() {
  awk '{print tolower($0);}' | sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
  -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
  -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
  -e 's/«/ /g' | tr 0-9 " "
}

wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.fr.shuffled.gz
wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.fr.shuffled.gz
gzip -d news.2012.fr.shuffled.gz
gzip -d news.2013.fr.shuffled.gz
normalize_text < news.2012.fr.shuffled > data.txt
normalize_text < news.2013.fr.shuffled >> data.txt
