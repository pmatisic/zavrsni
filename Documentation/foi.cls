\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{foi}

\newcommand{\jezik}{croatian}

\DeclareOption{english}{\renewcommand{\jezik}{english}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}


\ProcessOptions
\LoadClass[11pt, onecolumn]{report}
\RequirePackage[a4paper, left=2.5cm, right=2.5cm, bottom=2.5cm, top=2.5cm]{geometry}
\RequirePackage[utf8]{inputenc}
\RequirePackage[\jezik]{babel}
\RequirePackage{csquotes}
\renewcommand*{\mkcitation}[1]{~#1}
\renewcommand\mkblockquote[4]{\enquote{#1#2#3}#4}

\RequirePackage[T1]{fontenc}
\RequirePackage{anyfontsize}
\RequirePackage{graphicx} % Uključeno jer je često korišteno
\RequirePackage{amssymb} % Uključeno jer je često korišteno
\RequirePackage{amsmath} % Uključeno jer je često korišteno
\RequirePackage{caption}
\RequirePackage{ifthen}
\RequirePackage{url} % Potrebno radi natbiba
\RequirePackage{enumitem} % Potrebno radi izmjene itemize okoline
\RequirePackage{titlesec}
\RequirePackage{tabularx} % tablice
\RequirePackage{fancyhdr} % zaglavlja
\RequirePackage{chngcntr} % promjena brojača
\RequirePackage[table]{xcolor} % bojanje tablica
\RequirePackage{listings} % ispis programskog koda
\RequirePackage{times}
\RequirePackage{courier} % za programski kod
\RequirePackage[backend=biber, style=ieee, date=terse, isbn=true, url=false]{biblatex} % IEEE stil citiranja
\addbibresource{lib.bib}

\RequirePackage{comment}
\RequirePackage{indentfirst}
\RequirePackage{wrapfig}
\RequirePackage{setspace}

% stvaranje poveznica u dokumentu
\RequirePackage[pdftex,
bookmarks=true,
bookmarksnumbered=true,
pdfpagemode=UseNone,
pdfstartview=FitH,
pdfpagelayout=SinglePage,
colorlinks=false,
urlcolor=black,
pdfborder={0 0 0},
unicode
]{hyperref} 

% postavke za programski kod s UTF8 znakovima
\lstset{ 
    escapeinside={\%*}{*)},
    extendedchars=false,
    inputencoding=utf8,
    numberbychapter=false,
    basicstyle=\footnotesize\ttfamily,
    breaklines=true,
    showstringspaces=false,
    breakindent=5pt,
    numbers=left,
    numberstyle=\tiny,
    literate={č}{{\v{c}}}1
             {ć}{{\'c}}1
             {ž}{{\v{z}}}1
             {š}{{\v{s}}}1
             {đ}{{\dj}}1
             {Č}{{\v{C}}}1
             {Ć}{{\'C}}1
             {Ž}{{\v{Z}}}1
             {Š}{{\v{S}}}1
             {Đ}{{\DJ}}1
}

\newcommand{\prijevod}[2]{%
    \iflanguage{croatian}{#1}{%
    \iflanguage{english}{#2}{}}%
}%

% prijevod opisa isječka koda
\renewcommand{\lstlistingname}{\prijevod{Isječak k\^{o}da}{Listing}}

% Promjena iz Times u Sans font
\renewcommand{\rmdefault}{\sfdefault}

% Podešavanje broja stranice u donji desni kut
\pagestyle{fancy} 
\fancyhf{}
\fancyfoot[R]{\thepage}
\fancypagestyle{plain}{%
    \renewcommand{\headrulewidth}{0pt}%
    \fancyhf{}%
    \fancyfoot[R]{\thepage}%
}


% Redefiniranje datuma
% \def\date#1{\gdef\@date{#1}}

% 1.5 prored (1.2 je faktor za množenje)
\linespread{1.2}
% Veličina indenta na svakom paragrafu
\setlength{\parindent}{1.25cm}
% Veličina praznog prostora nakon svakog paragrafa
\setlength{\parskip}{6pt}

% Podešavanje oznaka kod listi (1. razina je crtica, 2. puni krug)
\renewcommand{\labelitemi}{\textbullet}
\renewcommand{\labelitemii}{\textendash}
\setlist{leftmargin=2cm}

% Slike numeriramo kroz cijeli dokument
\counterwithout{figure}{chapter}
\counterwithout{table}{chapter}
% \AtBeginDocument{\renewcommand{\thelstlisting}{\arabic{lstlisting}}}

% http://www.tex.ac.uk/cgi-bin/texfaq2html?label=seccntfmt
% dodaje tocku nakon section broja ali ne i nakon chapter!
% za chapter se koristi jednostavno http://theoval.cmp.uea.ac.uk/~nlct/latex/thesis/node10.html
\renewcommand*{\@seccntformat}[1]{%
  \csname the#1\endcsname
  \csname adddot@#1\endcsname.\quad
}
% Postavka točkica u sadržaju
\renewcommand*\l@chapter[2]{\textbf{\@dottedtocline{1}{0em}{2.3em}{#1}{#2}}}

%Dodane točke nakon brojeva u TOC
\let \savenumberline \numberline
\def \numberline#1{\savenumberline{#1.}}

% Redefinicija sadržaja
\let \originaltoc \tableofcontents
\renewcommand\tableofcontents{%
%\setcounter{page}{2}
\setcounter{secnumdepth}{3} % dubina do subsubsekcije
\setcounter{tocdepth}{3} % dubina do subsubsekcije
\chapter*{ \vspace{-7.5mm}\sffamily\fontsize{18pt}{0pt}\bfseries{\prijevod{Sadr\v zaj}{Table of Contents}}
\vspace{-10pt}
  \@mkboth{\MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
\begin{spacing}{0.6}
\@starttoc{toc}%
\end{spacing}
}

\def\musko{M}
\def\zensko{Z}
\def\odaberi{O}

\newcommand{\autor}[1]
{
  \ifthenelse{\equal{#1}{M}}{\prijevod{Autor}{author}\!\!\!\!\!\!}{}
  \ifthenelse{\equal{#1}{Z}}{\prijevod{Autorica}{author}\!\!\!\!\!\!}{}
  \ifthenelse{\equal{#1}{O}}{Autor/Autorica\!\!\!\!\!\!}{}
}

\newcommand{\mentorr}[1]
{
  \ifthenelse{\equal{#1}{M}}{\!\!\!\!Mentor}{}%
  \ifthenelse{\equal{#1}{Z}}{\!\!\!\!\!\prijevod{Mentorica}{Mentor}}{}%
  \ifthenelse{\equal{#1}{O}}{\!\!\!\!\!\!\!Mentor/Mentorica}{}%
}

\newcommand{\potvrdio}[1]
{
  \ifthenelse{\equal{#1}{M}}{potvrdio\!\!\!\!\!}{}
  \ifthenelse{\equal{#1}{Z}}{potvrdila\!\!\!}{}
  \ifthenelse{\equal{#1}{O}}{potvrdio/potvrdila\!}{}
}

% Podešavanje poglavlja tako da se ne ispisuje "Poglavlje <broj>"
\newboolean{prvi}
\setboolean{prvi}{false}
\newboolean{drugi}
\setboolean{drugi}{false}
\renewcommand{\@makechapterhead}[1]{%

%%
\ifthenelse{\boolean{prvi}}{%
	{\setlength{\parskip}{0em} \parindent \z@ \raggedright \normalfont
		\interlinepenalty\@M
		\ifnum \c@secnumdepth >\m@ne
			\par\nobreak
		\fi
		\nobreak
		\vskip 15\p@
	}
}
%%
%%
{
\ifthenelse{\boolean{drugi}}

	{
		\setcounter{page}{1}
		\renewcommand{\thepage}{\arabic{page}}
		\setboolean{drugi}{true}
	}


	{
                     \setlength{\parskip}{0em} \parindent \z@ \raggedright \normalfont
		\interlinepenalty\@M
		\ifnum \c@secnumdepth >\m@ne
			\sffamily\fontsize{18pt}{18pt}\bfseries \thechapter.~#1
			\par\nobreak
		\fi
		\nobreak
		\vskip 15\p@
	}
}
%%
}

% Oblici za podnaslove
\titleformat{\section}
  {\normalfont\fontsize{16pt}{16pt}\sffamily\bfseries}{\thesection.}{1em}{}
\titleformat{\subsection}
  {\normalfont\fontsize{14pt}{14pt}\sffamily\bfseries}{\thesubsection.}{1em}{}
\titleformat{\subsubsection}
  {\normalfont\fontsize{12pt}{12pt}\sffamily\bfseries}{\thesubsubsection.}{1em}{}

% Definicija varijabilnih vrijednosti
\def\vrstaRada#1{\gdef\@vrstaRada{\uppercase{#1}}}
\def\title#1{\gdef\@title{\uppercase{#1}}}
\def\predmet#1{\gdef\@predmet{\uppercase{#1}}}
\def\mentor#1{\gdef\@mentor{#1}}
\def\godina#1{\gdef\@godina{#1}}
\def\mjesec#1{\gdef\@mjesec{#1}}
\def\status#1{\gdef\@status{#1}}
\def\indeks#1{\gdef\@indeks{#1}}
\def\smjer#1{\gdef\@smjer{#1}}
\def\titulaProfesora#1{\gdef\@titulaProfesora{#1}}
\def\sazetak#1{\gdef\@sazetak{#1}}
\def\kljucneRijeci#1{\gdef\@kljucneRijeci{#1}}
\def\spolStudenta#1{\gdef\@spolStudenta{#1}}
\def\spolMentora#1{\gdef\@spolMentora{#1}}

\def\zavrsni{\prijevod{završni rad}{Bachelor's Thesis}}
\def\diplomski{\prijevod{diplomski rad}{Master's Thesis}}
\def\projekt{\prijevod{projekt}{Project}}
\def\seminar{\prijevod{seminar}{Seminar Paper}}

\def\predmetTBP{\prijevod{Teorija baza podataka}{Theory of Databases}}
\def\predmetVAS{\prijevod{Višeagentni sustavi}{Multiagent Systems}}
\def\predmetDP{\prijevod{Deklarativno programiranje}{Declarative Programming}}
\def\predmetBP{\prijevod{Baze podataka 1}{Databases 1}}

% Naslovne stranice
\renewcommand\maketitle{%
%
    \thispagestyle{empty}
    \begin{center}
      {\large \fontsize{14}{17}\selectfont\bfseries{\prijevod{SVEU\v CILI\v STE U ZAGREBU}{UNIVERSITY OF ZAGREB}}} \\[8pt]
      {\large \fontsize{14}{17}\selectfont\bfseries{\prijevod{FAKULTET ORGANIZACIJE I INFORMATIKE}{FACULTY OF ORGANIZATION AND INFORMATICS}}} \\[8pt]
      {\large \fontsize{14}{17}\selectfont\bfseries{VARA\v ZDIN}}
    \end{center}
    
    \vskip 20mm
    \begin{flushleft}
    \Large \bfseries{\@author}
    \end{flushleft}
    
    \vskip 60mm
    \begin{center}
    { \linespread{2.5}\sffamily\bfseries{\fontsize{22}{27}\selectfont\MakeUppercase\@title}\par}\ \\[10mm]
    {\normalfont \large \bfseries{\fontsize{14}{17}\selectfont\MakeUppercase\@vrstaRada}}\ \\[5mm]
    { \linespread{2.5}\sffamily\bfseries{\fontsize{22}{27}\selectfont\MakeUppercase\@predmet}\par}
    % {\normalfont \large \bfseries{\fontsize{14}{17}\selectfont\MakeUppercase\@predmet}}
    \end{center}
    
    \vfill
    \begin{center}
    {\large \bfseries{Vara\v zdin}, \bfseries{\@godina}\prijevod{.}{}}
    \end{center}
    
    \newpage
    
    \thispagestyle{empty}
    \begin{center}
      {\bfseries{\prijevod{SVEU\v CILI\v STE U ZAGREBU}{UNIVERSITY OF ZAGREB}}} \\[12pt]
      {\bfseries{\prijevod{FAKULTET ORGANIZACIJE I INFORMATIKE}{FACULTY OF ORGANIZATION AND INFORMATICS}}} \\[12pt]
      {\bfseries{V A R A \v Z D I N}}
    \end{center}
    
    \vskip 15mm
    \begin{flushleft}
    {\bfseries{\@author}}\\[8pt]
    {\bfseries{\prijevod{Matični broj}{Student ID}:} \bfseries{\@indeks}}\\[8pt]
    {\bfseries{\prijevod{Studij}{Programme}:} \bfseries{\@smjer}}\\[8pt]
    \end{flushleft}
    
    \vskip 40mm
    \begin{center}
    {\setstretch{1.2}\sffamily \bfseries{\fontsize{14}{17}\selectfont\MakeUppercase\@title}\par} \vspace{10mm}
    {\normalfont \bfseries{\MakeUppercase\@vrstaRada}}
    \end{center}
    
    \vskip 20mm
    \begin{flushleft}
    \hspace{90mm}
    \iflanguage{croatian}{%
        {\bfseries{\mentorr{\@spolMentora}}:}\\[15pt]%
    }{%
        \iflanguage{english}{%
            Mentor:\\[15pt]
        }{}
    }
    \hspace{90mm}%
    {\normalfont {\@titulaProfesora}}
    {\normalfont {\@mentor}}%
    \prijevod{}{, PhD}
    \end{flushleft}
    
    
    \vfill
    \begin{center}
    {\bfseries{Vara\v zdin}, \bfseries{\@mjesec} {\@godina}\prijevod{.}{}}
    \end{center}
    
    \pagenumbering{roman}
    \setcounter{page}{0}
    \newpage
    \thispagestyle{plain}
    
    \begin{flushleft}
    \textit{\@author}
    \end{flushleft}
    
    {\begin{center}\bfseries{\prijevod{Izjava o izvornosti}{Statement of Authenticity}}\end{center}}
    
    \prijevod{
    
        \noindent Izjavljujem da je ovaj \@vrstaRada\ izvorni rezultat mojeg rada te da se u izradi istoga nisam koristio drugim izvorima osim onima koji su u njemu navedeni. Za izradu rada su korištene etički prikladne i prihvatljive metode i tehnike rada.
        
        \begin{flushright}
        \textit{\autor{\@spolStudenta} \potvrdio{\@spolStudenta} prihvaćanjem odredbi u sustavu FOI Radovi}\par
        \end{flushright}
        
    }{
    
        \noindent Hereby I state that this document, my \@vrstaRada, is authentic, authored by me, and that, for the purposes of writing it, I have not used any sources other than those stated in this thesis. Ethically adequate and acceptable methods and techniques were used while preparing and writing this thesis.
        
        \begin{flushright}
        \textit{The author acknowledges the above by accepting the statement in FOI Radovi online system.}\par
        \end{flushright}
    }
    
    \noindent\makebox[\linewidth]{\rule{\textwidth}{0.4pt}}
    
    \newpage
    \thispagestyle{plain}
    
    {\begin{center}\bfseries{\prijevod{Sažetak}{Abstract}}\end{center}}
    
    \noindent \@sazetak
    
    \begin{flushleft}
    \textbf{\prijevod{Ključne riječi}{Keywords}}: \@kljucneRijeci
    \end{flushleft}


}

\newif\iffigures
\newif\iftables
\newif\iflistings

\makeatletter
\AtEndDocument {%
    \if@filesw
       \ifnum\value{figure}=\z@ % no figures
          \immediate\write\@mainaux {\global\string\figuresfalse}%
       \else
          \immediate\write\@mainaux {\global\string\figurestrue}%
       \fi
       \ifnum\value{table}=\z@ % no tables
          \immediate\write\@mainaux {\global\string\tablesfalse}%
       \else
          \immediate\write\@mainaux {\global\string\tablestrue}%
       \fi
       \ifnum\value{lstlisting}=\z@ % no listings
          \immediate\write\@mainaux {\global\string\listingsfalse}%
       \else
          \immediate\write\@mainaux {\global\string\listingstrue}%
       \fi
    \fi
}
\makeatother 

\newcommand{\makebackmatter}{
    \printbibliography[title=\prijevod{Popis literature}{Bibliography}]
    \addcontentsline{toc}{chapter}{\prijevod{Popis literature}{Bibliography}}
    
    % 
    \iffigures
       \listoffigures
       \addcontentsline{toc}{chapter}{\prijevod{Popis slika}{List of Figures}}
    \fi
    
    % 
    \iftables
       \listoftables
       \addcontentsline{toc}{chapter}{\prijevod{Popis tablica}{List of Tables}}
    \fi
    
    \iflistings
        \renewcommand{\tableofcontents}{\originaltoc}
        \def\@dotsep{4.5}
        \renewcommand{\lstlistlistingname}{\prijevod{Popis isječaka koda}{List of Listings}}
        \lstlistoflistings
        \addcontentsline{toc}{chapter}{\prijevod{Popis isječaka koda}{List of Listings}}
        \sectionmark{\prijevod{Popis isječaka koda}{List of Listings}}
    \fi
}

\newcommand{\appendices}{
    \appendix
    \clearpage
    \thispagestyle{empty}
    \hspace{0pt}
    \vfill
    \begin{center}
        \textbf{\prijevod{Prilozi}{Appendices}}
    \end{center}
    \vfill
    \hspace{0pt}
    
    \renewcommand{\thechapter}{\arabic{chapter}}
}