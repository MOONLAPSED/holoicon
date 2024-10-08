# holoicon for obsidian and ollama+ on ubuntu-22.04 & win11

## associative knowledge base (this repo):
All directories which contain markdown files are to include a `/media/` sub directory for multimedia files the markdown files may reference.

To enable horrors such as this:

![this:](/media/image.png)

    `! [ ... ] ( /media/image.png )` (no spaces)

 - [obsidian-markdown and associative 'knowledge base' README](/src/obs/README.md)

## Frontmatter Implementation

 - Utilize 'frontmatter' to include the title and other `property`, `tag`, etc. in the knowledge base article(s).
   
   - For Example:
      ```
      ---
      name: "Article Title"
      link: "[[Related Link]]"
      linklist:
        - "[[Link1]]"
        - "[[Link2]]"
      ---
      ``` 

## `README.md` glossary:
