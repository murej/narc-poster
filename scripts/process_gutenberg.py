#
#   Extract all book titles in a target language from project Gutenberg:
#   https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.zip
#
#   HOW TO: Unzip and dump the files in /data folder, then run this script
#

import os
import xmltodict

targetLanguage = 'cn'

dir = os.path.dirname(__file__)

dataPath = os.path.join(dir, '../data')
outputFilename = 'titles_' + targetLanguage + '.txt'
outputPath = os.path.join(dir, '../data/' + outputFilename)

output = open(outputPath, 'w')
output.truncate()

extractedTitleNr = 0


def isEnglish(ebook):
    if 'dcterms:language' in ebook:
        langList = []
        langOrLangs = ebook['dcterms:language']

        # if single language
        if isinstance(langOrLangs, list):
            langList = langOrLangs
        else:
            langList.append(langOrLangs)

        for langItem in langList:
            lang = langItem['rdf:Description']['rdf:value']
            lang = list(lang.items()[1])[1]

            # if english found
            if lang == targetLanguage:
                return True


def writeTitle(ebook, output):
    if 'dcterms:title' in ebook:
        titleOrTitles = ebook['dcterms:title']
        if isinstance(titleOrTitles, list):
            title = titleOrTitles[0]
        else:
            title = titleOrTitles
        title = title.encode('utf-8').strip().replace('\n', ' - ').replace('\r', '')
        output.write(title)
        output.write("\n")
        return 1
    else:
        return 0


for path, subDirs, fileNames in os.walk(dataPath):
    for fileName in fileNames:
        relativePath = os.path.join(path, fileName)
        print relativePath
        if relativePath.endswith('.rdf'):
            with open(relativePath) as fd:
                content = xmltodict.parse(fd.read())
                ebook = content['rdf:RDF']['pgterms:ebook']
                if isEnglish(ebook):
                    extractedTitleNr += writeTitle(ebook, output)

output.close()
print "Done! Extracted " + str(extractedTitleNr) + " titles to " + outputFilename
