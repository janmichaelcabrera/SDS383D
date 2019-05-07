% ----------------------------------------------------------------------
% -- Class definition
% ----------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\DeclareOption{preprint}{\AtBeginDocument{\doublespacing}}
\DeclareOption{authoryear}{\xdef\@biboptions{square,authoryear}}
\DeclareOption{review}{\AtBeginDocument{\pagewiselinenumbers \doublespacing}}
\DeclareOption{equation}{\AtBeginDocument{\numberwithin{equation}{section}}} %numbering equation as section
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass[onecolumn,twoside,a4,titlepage,epsf,ceqno]{article}
% ----------------------------------------------------------------------
% -- Packages
% ----------------------------------------------------------------------
\RequirePackage{latexsym}
\RequirePackage{amssymb,amsfonts,amsmath,amsthm}
\RequirePackage[dvips]{graphicx,epsfig}
\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\RequirePackage{fancyhdr}
\RequirePackage{sectsty}
\usepackage{appendix}
\usepackage{datetime}
\usepackage{algorithm}
\usepackage[noend]{algpseudocode}
\usepackage{amsbsy}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{caption}
\usepackage[color]{changebar}
\usepackage{datetime}
\usepackage{epsfig}
\usepackage{enumerate}
\usepackage{enumitem}
\usepackage{fancybox,amssymb}
\usepackage{fancyhdr}
\usepackage{floatflt}
\usepackage[OT1]{fontenc}
\usepackage{graphicx}
\usepackage[dvips]{graphicx}
\usepackage{latexsym}
\usepackage{lineno}
\usepackage{longtable}
\usepackage{multirow}
\usepackage[sort&compress,numbers]{natbib}
\usepackage[nice]{nicefrac}
\usepackage{overpic}
\usepackage{pstricks}
\usepackage{setspace}
\usepackage{stfloats}
\usepackage{subfigure,amsmath}
\usepackage{threeparttable}
\usepackage{type1cm}
\usepackage{type1ec}
\usepackage{textfit}
\usepackage{upgreek}

\cbcolor{black}


% ----------------------------------------------------------------------
% -- Page settings
% ----------------------------------------------------------------------
\pagestyle{fancy}


%\spaceskip .15cm\relax
%\kern-0.19cm
%\hsize=100in
%\rightskip=0pt plus2em\spaceskip=.19cm\xspaceskip=.8em\relax
%\space 0.19cm

\linespread{1}

\bfseries
%\sloppy
%\hyphenpenalty=10000
%\clubpenalty=10000
%\widowpenalty=10000
\brokenpenalty=10000	
\oddsidemargin=-5.2mm
\evensidemargin=-5.2mm
%\setlength{\columnsep}
\setlength{\textheight}{229mm}
%\setlength{\textheight}{720pt}
%\footskip 100pt
\setlength{\textwidth}{40.1pc}
%\spaceskip=3pt plus .333em\rightskip0pt
%\frenchspacing
\voffset -25mm %
% ----------------------------------------------------------------------
% -- Page settings
% ----------------------------------------------------------------------


% ----------------------------------------------------------------------
% -- Title
% ----------------------------------------------------------------------
%\providecommand{\address}[1]{\date{\small\it#1}}
%\providecommand{\email}[2]{\footnote[0]{\sups{#1}\small\texttt{#2}}}
%\def\authorhead#1{\gdef\@authorhead{#1}}
%\def\titlehead#1{\gdef\@titlehead{#1}}
%\def\address#1{\gdef\@address{#1}}
%\def\email#1{\gdef\@email{#1}}
\def\dataO#1{\gdef\@dataO{#1}}
\def\dataF#1{\gdef\@dataF{#1}}
\def\volume#1{\gdef\@volume{#1}}


\let\comma\@empty
\let\tnotesep\@empty
\def\title#1{\gdef\@title{#1}}
\let\@title\@empty

\def\begLabel#1{\@bsphack\protected@write\@auxout{}%
         {\string\Newlabel{#1}{\@currentlabel}}\@esphack}
\def\Newlabel#1#2{\expandafter\xdef\csname X@#1\endcsname{#2}}

\def\begRef#1{\@ifundefined{X@#1}{0}{\csname X@#1\endcsname}%
}

\def\elsRef#1{\@ifundefined{X@#1}{0}{\csname X@#1\endcsname}%
}

\def\tnotemark[#1]{\textsuperscript{\@for\@@tmark:=#1\do{%
      \edef\tnotenum{\@ifundefined{X@\@@tmark}{1}{\elsRef{\@@tmark}}}%
      \ifcase\tnotenum\or\ding{73}\or,\ding{73}\ding{73}\fi}}%
}
\let\@tnotemark\@empty

\let\@tnotes\@empty
\RequirePackage{pifont}
\newcounter{tnote}
\def\tnotetext[#1]#2{\g@addto@macro\@tnotes{%
  \refstepcounter{tnote}\elsLabel{#1}%
   \def\thefootnote{\ifcase\c@tnote\or\ding{73}\or\ding{73}\ding{73}\fi}%
    \footnotetext{#2}}}

\let\@nonumnotes\@empty
\def\nonumnote#1{\g@addto@macro\@nonumnotes{%
     \let\thefootnote\relax\footnotetext{#1}}}

\newcounter{fnote}
\def\fnmark[#1]{\let\comma\@empty
    \def\@fnmark{\@for\@@fnmark:=#1\do{%
    \edef\fnotenum{\@ifundefined{X@\@@fnmark}{1}{\elsRef{\@@fnmark}}}%
  \unskip\comma\fnotenum\let\comma,}}%
}

\let\@fnotes\@empty\let\@fnmark\@empty
\def\fntext[#1]#2{\g@addto@macro\@fnotes{%
     \refstepcounter{fnote}\elsLabel{#1}%
   \def\thefootnote{\thefnote}%
   \global\setcounter{footnote}{\thefnote}%
   \footnotetext{#2}}}

\def\cormark[#1]{\edef\cnotenum{\elsRef{#1}}%
    \unskip\textsuperscript{\sep\ifcase\cnotenum\or
       $\ast$\or$\ast\ast$\fi\hspace{-1pt}}\let\sep=,}

\let\@cormark\@empty
\let\@cornotes\@empty
\newcounter{cnote}
\def\cortext[#1]#2{\g@addto@macro\@cornotes{%
    \refstepcounter{cnote}\elsLabel{#1}%
    \def\thefootnote{\ifcase\thecnote\or$\ast$\or
    $\ast\ast$\fi}%
    \footnotetext{#2}}}

\let\@corref\@empty
\def\corref#1{\edef\cnotenum{\elsRef{#1}}%
    \edef\@corref{\ifcase\cnotenum\or
         $\ast$\or$\ast\ast$\fi\hskip-1pt}}

\def\fnref#1{\fnmark[#1]}
\def\tnoteref#1{\tnotemark[#1]}

\def\resetTitleCounters{\c@cnote=0
   \c@fnote=0 \c@tnote=0 \c@footnote=0}



\let\emailsep\@empty
\let\@allNonCorrEmail\@empty        %%kk
\let\@beguads\@empty
\let\@cormark\@empty
\def\hashchar{\expandafter\@gobble\string\~}
\def\underscorechar{\expandafter\@gobble\string\_}
\def\lbracechar{\expandafter\@gobble\string\{}
\def\rbracechar{\expandafter\@gobble\string\}}

\def\email{\@ifnextchar[{\@uad}{\@email}}
\gdef\@email#1{\bgroup\def\_{\string\underscorechar\space}%
  \def\{{\string\lbracechar\space}%
   \def~{\hashchar\space}%
   \def\}{\string\rbracechar\space}%
   \edef\tmp{\the\@emailauthor}
   \immediate\write\@auxout{\string\emailauthor
     {#1}{\expandafter\strip@prefix\meaning\tmp}}%
  \egroup
}


%\newcounter{email}
%\gdef\emailcorrauthor#1#2{\stepcounter{email}%
%     \g@addto@macro\@allemail{\raggedright%
%      \let\corref\@gobble
%      \emailsep{#2} E-mail: #1\def\emailsep{\unskip \space\space and \space}}%
      %\emailsep E-mail: #1\def\emailsep{\unskip \space\space and \space}}%
%}
\newcounter{email}
\gdef\emailauthor#1#2{\stepcounter{email}%
     \g@addto@macro\@allNonCorrEmail{\raggedright%      %%sk
      \let\corref\@gobble
      \emailsep{#2} E-mail: #1\def\emailsep{\unskip \space\space and \space}}%
      %\emailsep E-mail: #1\def\emailsep{\unskip \space\space and \space}}%
}

\def\begauthors{}
\def\pprinttitle{}
\let\authorsep\@empty
\let\sep\@empty
\newcounter{author}
\def\author{\stepcounter{author}\@ifnextchar[{\@@author}{\@author}}
\newcounter{corrauthor}
\def\corrauthor{\@ifnextchar[{\@@corrauthor}{\@corrauthor}}


%\tracingmacros=1
\def\authorCommaList{}
\let\authorAndSep\@empty

\newtoks\@emailauthor
\def\@@author[#1]#2{
    \let\begauthors=\authorCommaList
    \g@addto@macro\begauthors{%%
        \def\baselinestretch{1}%%
        \gdef\tempSuperscript{\textsuperscript{\@for\@@affmark:=#1\do{\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}\unskip\sep\affnum\let\sep=,}}}
        \authorAndSep#2\unskip\tempSuperscript%\textsuperscript{%%
            %\@for\@@affmark:=#1\do{%
            %\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            %\unskip\sep\affnum\let\sep=,}%
         %}%%
    }
    \g@addto@macro\authorCommaList{%
         \def\baselinestretch{1}%%
         \authorsep#2\unskip\comma\textsuperscript{%%
            \@for\@@affmark:=#1\do{%
            \edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            \unskip\sep\affnum\let\sep=,}%%
         }\space %%
        %\def\authorsep{\unskip, \space}%%
        \def\authorAndSep{\unskip \space}%%
    }
    \@emailauthor={#2}
}

\def\@author#1{
    \let\begauthors=\authorCommaList
    \g@addto@macro\begauthors{%%
        \def\baselinestretch{1}%%
        \authorAndSep#1\unskip\textsuperscript{%%
            %\@for\@@affmark:=#1\do{%
            %\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            %\unskip\sep $^*$\let\sep=,%}%
         }%%
    }
    \g@addto@macro\authorCommaList{%
         \def\baselinestretch{1}%%
         \authorsep#1\unskip\comma\textsuperscript{%%
            %\@for\@@affmark:=#1\do{%
            %\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            \unskip\sep \let\sep=,%}%%
         }\space %%
        %\def\authorsep{\unskip, \space}%%
        \def\authorAndSep{\unskip \space}%%
    }
    \@emailauthor={#1}
}

\def\authorsep{%
\ifnum \c@author = 2%
   \def\comma{}%
      \else%
    \def\comma{,}%
     \fi}


%\tracingmacros=1

%%%kk

\def\corrAuthorName{}
\def\corrAuthorEmail{}
\def\@@corrauthor[#1]#2{\stepcounter{author}%fix for counter author
    %\author[#1]{#2}
        \let\begauthors=\authorCommaList
    \g@addto@macro\begauthors{%%
        \def\baselinestretch{1}%%
        \authorAndSep#2\unskip\textsuperscript{%%
            \@for\@@affmark:=#1\do{%
            \edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            \unskip\sep\affnum,$*$\let\sep=,}%
         }%%
    }
     \g@addto@macro\authorCommaList{%
         \def\baselinestretch{1}%%
         \authorsep#2\unskip\comma\textsuperscript{%%
            \@for\@@affmark:=#1\do{%
            \edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            \unskip\sep\affnum,$*$\let\sep=,}%%
         }\space %%
        %\def\authorsep{\unskip, \space}%%
        \def\authorAndSep{\unskip\space \& \space}%%
    }
    \@emailauthor={#2}

    \def\corrAuthorName{#2}     % use |corrAuthorName|
}

\def\@corrauthor#1{\stepcounter{author}%fix for counter author
        \let\begauthors=\authorCommaList
    \g@addto@macro\begauthors{%%
        \def\baselinestretch{1}%%
        \authorAndSep#1\unskip\textsuperscript{%%
            %\@for\@@affmark:=#1\do{%
            %\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            %\unskip\sep\affnum,$*$\let\sep=,}%
            $*$
         }%%
    }
    \g@addto@macro\authorCommaList{%
         \def\baselinestretch{1}%%
         \authorsep#1\unskip\comma\textsuperscript{%%
            %\@for\@@affmark:=#1\do{%
            %\edef\affnum{\@ifundefined{X@\@@affmark}{1}{\begRef{\@@affmark}}}%
            %\unskip\sep\affnum,$*$\let\sep=,}%%
            $*$
         }\space %%
        %\def\authorsep{\unskip, \space}%%
        \def\authorAndSep{\unskip\space \& \space}%%
    }
    \@emailauthor={#1}
    \def\corrAuthorName{#1}     % use |corrAuthorName|
}



%    \author{#1}
%    \def\corrAuthorName{#1}     % use |corrAuthorName|


\def\corremail#1{
    \def\corrAuthorEmail{#1}     % use |corrAuthorEmail|
}
\def\corrAuthorUrl{}
\def\corrurl#1{
    \def\corrAuthorUrl{#1}     % use |corrAuthorUrl|
}

\def\addallcorem{}
\def\addallcor#1{
    \def\addallcorem{#1}     % use |addallcorem|
}

\def\phoneAuthor{}
\def\phone#1{
    \def\phoneAuthor{#1}
}
\def\faxAuthor{}
\def\fax#1{
    \def\faxAuthor{#1}
}

\def\doiart{}
\def\doi#1{
    \def\doiart{#1}
}

\def\dataallf{}
\def\dataF#1{
    \def\dataallf{#1}     % use |corrAuthorUrl|
}
\def\dataallo{}
\def\dataO#1{
    \def\dataallo{#1}     % use |corrAuthorUrl|
}

\def\dateSub{\ifx\dataallo\@empty\else{\rm {\it{\Large{\fontfamily{upl}\selectfont\small   Original Manuscript Submitted: \dataallo; Final Draft Received: \dataallf}}}
  \vskip 1.5pc}
  \fi}



\def\addressCorrespondenceText{~\\[19pt]\parbox{40.4pc}{
    \small$^*$Corresponding %S. Mahadevan,
          \corrAuthorName: \corrAuthorEmail\phoneAuthorText\faxAuthorText%\corrAuthorUrlText
           \newline
           % \doiartT\\[9pt]}  %%kk
           \doiartT\\[-6pt]}  %%kk
    }

%\def\corrAuthorUrlText{%
%       \ifx\corrAuthorUrl\@empty\else{\small,\space
% URL: \corrAuthorUrl } \fi% %kk
%    }

 \def\phoneAuthorText{%
       \ifx\phoneAuthor\@empty\else{\small;\space
 phone: \phoneAuthor}\fi% %kk
}

 \def\faxAuthorText{%
       \ifx\faxAuthor\@empty\else{\small;\space
 fax: \faxAuthor}\fi% %kk
}

 \def\doiartT{%
       \ifx\doiart\@empty
       \else{\normalsize DOI: \doiart}\fi% %kk
}

\def\addressOtherCorrespondenceText{
       \ifx\@allNonCorrEmail\@empty \else {\newline
\small Other address  \@allNonCorrEmail \newline\newline}\fi% %kk
}
%%%kk


\newcommand\fottitle{
\scriptsize{\addressCorrespondenceText{}}
}


\def\titleConf#1{\gdef\@titleConf{#1}}
\def\city#1{\gdef\@city{#1}}


\def\abrv#1{\gdef\@abrv{#1}}
\def\PaperNumber#1{\gdef\@PaperNumber{#1}}
\newcommand\headR{
\vspace*{-10pt}\small\textbf{\@titleConf}\hspace*{-123pt}\\ \small\textbf{\@city}\hspace*{-123pt}
}



\def\addvsp{}
\def\probel#1{
    \def\addvsp{#1}     % use |addvspace|
}

 \def\addprobbel{%
       \ifx\@addvsp\@empty\else{\addvsp} \fi% %kk
}



\def\title#1{
\uppercase{\def\uptitle{#1}}
}

 \def\addprobbel{%
       \ifx\@addvsp\@empty\else{\addvsp} \fi% %kk
}


\def\begaddress{}
\def\addsep{\par\vskip10pt}
\def\address{\@ifnextchar[{\@@address}{\@address}}

\def\@Roman#1{%
  \ifcase#1\or 1\or 2\or 3\or 4\or 5\or 6\or 7\or 8\or 9\or 10\or 11\or 12
    \else\@ctrerr\fi}

\newcounter{affn}
\renewcommand\theaffn{\Roman{affn}}

\long\def\@@address[#1]#2{\g@addto@macro\begaddress{%
    \def\baselinestretch{1}%
     \refstepcounter{affn}
     \xdef\@currentlabel{\theaffn}
     \begLabel{#1}%
     \begin{enumerate}[labelsep=-17pt, align=left, itemsep=0pt,
parsep=0pt, topsep=0pt, partopsep=0pt]\centering\item[\begLabel{#1}%
    \textsuperscript{\theaffn}]#2\end{enumerate}\vspace{0pt}\par}\vspace{55pt}}

%\long\def\@address#1{\g@addto@macro\begauthors{%
%    \def\baselinestretch{1}%
%    \addsep\footnotesize\itshape#1\def\addsep{\par\vskip6pt}%
%    \def\authorsep{\par\vskip8pt}}}

\long\def\@address#1{\g@addto@macro\begaddress{%
    %\def\baselinestretch{1}%
      %\xdef\@currentlabel{\theaffn}
     %\begLabel{#1}%
    %\textsuperscript{\theaffn}
    #1\addsep\par}}


  \setlength{\changebarsep}{70pt}

\def\cb@positions{%
  \global\cb@odd@left=\hoffset
  \global\cb@even@left\cb@odd@left
  \global\advance\cb@odd@left by \oddsidemargin
  \global\cb@odd@right\cb@odd@left
  \global\advance\cb@odd@right by \textwidth
  \global\advance\cb@odd@right by -\changebarsep
  \global\advance\cb@odd@right by -0.5\changebarwidth
  \global\advance\cb@odd@left by \changebarsep
  \global\advance\cb@odd@left by 0.5\changebarwidth

}
%\gdef\@authorhead{\texttt{no authorhead given}}






% -- new maketitle
%\topmargin -9pt
%\headsep 20pt
\renewcommand\maketitle{
\clearpage
   \thispagestyle{plain}
  \let\footnotesize\small
  %\let\footnoterule\relax
  %\let \footnote \thanks
\setlength{\textheight}{241mm}
\setlength{\textwidth}{40.1pc}

%\headsep 3pc
%\vspace*{60pt}
\parindent=0pt
\setlength{\parskip}{8pt plus 1pt minus 1pt}
%  \begin{center}%

{
%\parbox{15.8cm}
 %\parbox{31pc}
{%\vspace*{9mm}
%\hskip -0.5cm
\begin{minipage}{1 \textwidth} %\begin{tabular}{|l}
  %\hskip 0.3cm
  \vspace{4pt}
   %\parbox{15cm}


%\vspace*{-17pt}
\hspace*{-5pt}%\parbox{395pt}
{
\begin{flushleft}
\vspace*{-36pt}\includegraphics[height=14.0mm]{LOGO-eps-converted-to.pdf}
\end{flushleft}


\begin{flushright}
\vspace*{-1.0mm}%
\vspace*{-52pt}%%%%%%
% \tiny\textbf{\fontfamily{phv}\selectfont\@titleConf}\\%
% \tiny\textbf{\fontfamily{phv}\selectfont\@city}\\[16.0pt]%%%%%%%%%%
\Large\textbf{\fontfamily{phv}\selectfont\@PaperNumber}
\end{flushright}

\vspace*{-1.5mm}
\begin{center}{\vskip 0.26cm\LARGE\textbf{\uptitle}}\end{center}
    \vspace*{-7pt}%
     {\begin{center}\centering{\centering\textbf{\begauthors}}\end{center}}
    \vspace*{1pt}%
        {{\begaddress}} }
\addprobbel

%\dateSub
%\vskip 0.65cm


\vspace{10pt}
% -- abstract
\newenvironment{@wqabs}
{\list{}{\listparindent -0.5cm%
         \itemindent \listparindent
         \leftmargin 0in%
         \rightmargin \leftmargin
         \parsep \z@ \@plus \p@}%
         \item\relax}
{\endlist}

\begin{@wqabs}
\begin{center}
\large{\textbf{ABSTRACT}}
\end{center}\vspace{15pt}
\hspace*{15pt}\parbox{478pt}
{
{\@abstract}
\vskip 0.4cm
\large{\textbf{KEY WORDS:}}
{\small\@keywords}\vspace*{5pt}
}
\end{@wqabs}



     %\end{tabular}
     %\vspace{10pt}
     \end{minipage}}
 }




\vskip 0pt
  \@thanks
 % \vfil\null

     \pagestyle{fancy}
     %\baselineskip=2
%
%\setlength{\columnsep}{1pc}
%\setlength{\textheight}{9in}
%\setlength{\textwidth}{40pc}
%\setlength{\textheight}{730pt}
\setlength{\textheight}{239mm}
\setlength{\textwidth}{40.1pc}
%\headsep 3pc
%\hfuzz2pc
%\topmargin 0in
\footskip 22pt
%\footskip 4pc

  %\end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\and\relax
}



\renewcommand{\abstract}[1]{\def\@abstract{#1}}
\providecommand{\keywords}[1]{\def\@keywords{#1}} %\topmargin 0in
%
%----------------------------------------------------------------------
% -- Headers and footers %
%----------------------------------------------------------------------
\renewcommand{\headrulewidth}{0pt} \fancyhf{}
\fancyhf[FCO]{\hrule height0pt\vspace*{19pt}\large\thepage\hspace*{-120pt}}% Odd
\fancyhf[FCE]{\hrule height0pt\vspace*{19pt}\large\thepage\hspace*{120pt}} % Even
\def\@hrdef{\vspace*{17pt}\large\fontfamily{phv}\selectfont\@PaperNumber}
% \def\@hrdef{\vspace*{0pt}\large\fontfamily{phv}\selectfont\@PaperNumber}
\fancyhf[HRO]{\@hrdef\hspace*{-135pt}\vspace*{-12pt}}% Odd
\fancyhf[HRE]{\@hrdef\vspace*{-12pt}}% Even
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \fancyhf[FRO]{\fontfamily{phv}\fontsize{12}{10pt}\selectfont\@abrv\hspace*{-120pt}}
% \fancyhf[FRE]{\fontfamily{phv}\fontsize{12}{10pt}\selectfont\@abrv}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\fancyhf[HRO]{\small{\bf\thepage}\hspace*{-36pt}}
%\fancyhf[HLE]{\hspace*{-36pt}\small{\bf\thepage}}
%\fancyhf[HRE]{\scriptsize{{\bf\textsc{ Annual Review of Heat Transfer}}}}
%\fancyhf[HLO]{\scriptsize{{\bf\textsc{\@titlehead}}}} %
% ---------------------------------------------------------------------------
\frenchspacing
\fancypagestyle{plain}{%
  \fancyhf{\relax}
  % \fancyhead[L]{\small \headL}
  % \fancyhead[R]{\fontsize{12pt}{10pt}{ \@PaperNumber}
  \fancyfoot[c]{\large\fottitle\large\thepage\hspace*{-120pt}}
  }
% ---------------------------------------------------------------------------


%----------------------------------------------------------------------
% -- Numbering %
%----------------------------------------------------------------------
%\numberwithin{equation}{section}
\renewcommand{\theenumi}{\Roman{enumi}}
%\renewcommand{\theenumii}{\roman{enumi}.\arabic{enumii}}
\renewcommand{\labelenumii}{\theenumii.}
\renewcommand{\labelitemi}{\labelitemii} %
\renewcommand{\labelitemi}{$\bullet$}
%----------------------------------------------------------------------
% -- Sectioning %
%----------------------------------------------------------------------
% \def\@seccntformat#1{\csname the#1\endcsname.\;\,}

\def\@seccntformat#1{\csname pre#1\endcsname\csname
the#1\endcsname
                \csname post#1\endcsname}
\let\@Tocseccntformat\@seccntformat
\def\@postskip@{\hskip.5em\relax} \def\postsection{.\@postskip@}
\def\postsubsection{.\@postskip@}
\def\postsubsubsection{.\@postskip@}
\def\postparagraph{\@postskip@}
\def\postsubparagraph{.\@postskip@} \def\presection{}
\def\presubsection{} \def\presubsubsection{} \def\preparagraph{}
\def\presubparagraph{}


% sect Align Left
\def\MTKPsectAlignLeft{%
  \let\\\@centercr\@rightskip\@flushglue \rightskip\@rightskip%
  \leftskip\z@skip}

\def\MTKPsectionStyle{\sf\normalsize\bfseries\uppercase}

\renewcommand\thesubsubsection {\thesubsection .\arabic{subsubsection}}

\renewcommand\section{\@startsection{section}{1}{\z@}%
                                   {3ex \@plus 1ex \@minus .2ex}%
                                   {0.5ex \@plus.2ex}%
                                   {\centering\large\bfseries\uppercase}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                   {-1.5ex \@plus -1ex \@minus -.2ex}%
                                   {0.5ex \@plus.2ex}%
                                     {\MTKPsectAlignLeft\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}%
  \z@{.5ex\@plus.7ex}{-.5em}%
  {\normalfont\itshape\underline}}



\renewcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \setcounter{equation}{0}
  \gdef\thefigure{\@Alph\c@section.\arabic{figure}}%
  \gdef\thetable{\@Alph\c@section.\arabic{table}}%
  \gdef\thesection{\appendixname\@Alph\c@section}%
  \@addtoreset{equation}{section}%
  \gdef\theequation{\@Alph\c@section.\arabic{equation}}%
}
\def\appendixname{Appendix }


\newcommand\acknowledgements{\section* {Acknowledgments}%
}

\renewcommand\TPTnoteSettings{%
 \setlength\leftmargin{.7em}%
 \setlength\labelwidth{.5em}%
 \setlength\labelsep{.2em}%
 \small
 \rightskip\tabcolsep \leftskip\tabcolsep
}


% ----------------------------------------------------------------------
% -- Mathematical environment
% ----------------------------------------------------------------------
% -- theorems & related topics
\theoremstyle{plain}
\newtheorem{theo}{Theorem}[section]
%\newtheorem{prop}[theo]{Proposition}
\newtheorem{lemm}[theo]{Lemma}
\newtheorem{coro}[theo]{Corollary}
% -- definitions & examples
\theoremstyle{definition}
\newtheorem{defi}{Definition}[section]
\newtheorem{axio}{Axiom}[section]
% -- remarks & axioms
\theoremstyle{remark}
\newtheorem{rema}{Remark}[section]
\newtheorem{exam}{Example}[section]
% -- algorithms
\newtheoremstyle{algostyle}% name
  {}%      Space above, empty = `usual value'
  {}%      Space below
  {}%         Body font
  {}%         Indent amount (empty = no indent, \parindent = para indent)
  {\bfseries}% Thm head font
  {}%        Punctuation after thm head
  {\newline}% Space after thm head: \newline = linebreak
  {\thmname{#1}\thmnumber{ #2.}{ \texttt{[#3]}}}% Thm head spec
\theoremstyle{algostyle}
\newtheorem{algo}{Algorithm}
%
\def\ten#1{{\oalign{$\bf #1$\crcr\hidewidth$\scriptscriptstyle\sim$\hidewidth}}}
\def\tenq#1{\ten{\ten{#1}}} % tenseur d'ordre 4
\def\dten#1{\dot\ten{#1}} % derivee tenseur d'ordre 2
%
%\renewcommand\vec[1]{{\bf \underline{#1}}}
\newcommand\vect[1]{\vec{#1}}
\def\dvec#1{\dot\vect{#1}} % derivee vecteur
\def\FE2{$\hbox{FE}^2$}
%
%
\def\espacev{\vbox to 5mm{\vfill\vfill}}
\def\fe2{{\smash{$\hbox{FE}^2$}}}
\def\trace#1{\hbox{\bf Tr}#1}
\def\ronde#1{{\cal #1}}
\def\trans#1{{}^t#1}
\def\transp#1{{}^t#1}
\def\norm#1{||#1||}
\def\rang#1{\hbox{\bf rang}#1}
%
%
\def\code#1{{\tt #1}}
\def\zebu{{\bf Z\'eBuLoN}\xspace}
%
%\def\double{\baselineskip=35pt}
%\def\simple{\baselineskip=35pt}
%
\def\ds{\displaystyle}
%
\def\bblref#1{\ref{#1}}
\def\zebu{Z\'eBuLoN}
% ----------------------------------------------------------------------
% -- Text macros
% ----------------------------------------------------------------------
\providecommand{\sups}[1]{\raisebox{1ex}{\tiny #1}}
\providecommand{\subs}[1]{\raisebox{-.5ex}{\tiny #1}}
\providecommand{\ie}{i.e.,\;}
\providecommand{\eg}{e.g.,\;}
\providecommand{\ca}{ca.\;}
\providecommand{\cf}{cf.\;}
\providecommand{\etc}{etc.\;}
    \renewcommand{\postsection}{.\;\,}
    \renewcommand{\postsubsection}{\;\,}
    \renewcommand{\postsubsubsection}{\;\,}
    \renewcommand{\tiny}{\fontsize{8.03}{9.636pt}\selectfont}
    \renewcommand{\scriptsize}{\fontsize{9.035}{10.836pt}\selectfont}
    \renewcommand{\small}{\fontsize{10.04}{12.048pt}\selectfont}
    \renewcommand{\normalsize}{\fontsize{11.04}{13.248pt}\selectfont}
    \renewcommand{\large}{\fontsize{12.045}{14.454pt}\selectfont}
    \renewcommand{\Large}{\fontsize{13.05}{15.66pt}\selectfont}
    \renewcommand{\LARGE}{\fontsize{14.055}{16.86pt}\selectfont}
    
    \renewcommand{\@makefnmark}{\mathsurround=0pt
$^{\@thefnmark}$}
\renewcommand{\@makefntext}[1]{\parindent=1em\noindent {\@makefnmark}#1}

% ----------------------------------------------------------------------
% -- Tables
% ----------------------------------------------------------------------
\captionsetup{labelfont=bf, figurename=Fig., tablename=Table, labelsep=space, font=normalsize}
\captionsetup[table]{singlelinecheck=true}
\renewcommand{\bibsep}{3.5pt}
    \renewcommand{\bibnumfmt}[1] {[\text{#1}] }
    %\renewcommand{\thefootnote}{*}
    \renewcommand{\refname} {\large\text REFERENCES}
\providecommand{\wstretch}[1]{\def\arraystretch{#1}}


% ----------------------------------------------------------------------
% -- Greek
% ----------------------------------------------------------------------
\renewcommand{\alpha}{\upalpha}
\renewcommand{\beta}{\upbeta}
\renewcommand{\gamma}{\upgamma}
\renewcommand{\delta}{\updelta}
\renewcommand{\epsilon}{\upepsilon}
\renewcommand{\varepsilon}{\upvarepsilon}
\renewcommand{\zeta}{\upzeta}
\renewcommand{\eta}{\upeta}
\renewcommand{\theta}{\uptheta}
\renewcommand{\vartheta}{\upvartheta}
\renewcommand{\iota}{\upiota}
\renewcommand{\kappa}{\upkappa}
\renewcommand{\lambda}{\uplambda}
\renewcommand{\mu}{\upmu}
\renewcommand{\nu}{\upnu}
\renewcommand{\xi}{\upxi}
\renewcommand{\pi}{\uppi}
\renewcommand{\varpi}{\upvarpi}
\renewcommand{\rho}{\uprho}
%\renewcommand{\varrho}{\upvarrho}
\renewcommand{\sigma}{\upsigma}
%\renewcommand{\varsigma}{\upvarsigma}
\renewcommand{\tau}{\uptau}
\renewcommand{\upsilon}{\upupsilon}
\renewcommand{\phi}{\upphi}
\renewcommand{\varphi}{\upvarphi}
\renewcommand{\chi}{\upchi}
\renewcommand{\psi}{\uppsi}
\renewcommand{\omega}{\upomega}

% ------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------
