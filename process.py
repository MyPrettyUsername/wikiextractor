import shlex
import sys

import WikiExtractor
import postprocessing


def run(path_to_dump, wiki_files_dir='wiki', path_to_res='res_wiki.csv', workers_num=3):
    # step #1 - process files via wikiextractor
    argv = shlex.split(f'-o {wiki_files_dir} --json --processes {workers_num} {path_to_dump}')
    sys.argv = [sys.argv[0]] + argv
    print(argv)
    WikiExtractor.main()

    # step #2 - postporcessing
    postprocessing.run(wiki_files_dir, path_to_res, workers_num)


if __name__ == '__main__':
    run(*sys.argv[1:])
