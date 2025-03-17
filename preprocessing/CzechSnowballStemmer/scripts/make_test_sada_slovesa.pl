#!/usr/bin/perl

# Na zaklade koncovek vzoru vygeneruje danym slovum vsechny jejich existujici
# tvary.
# Jednotlive vzory se nachazi ve scripts/koncovky_sklonovani.
# Vstup:  slova s pridelenymi vzory (scripts/test_sada_slovesa.txt)
# Vystup: vsechny varianty slov ze vstupniho souboru (inputs/test_slovesa.txt)
#
# struktura vstupniho souboru:
# :muž                #znakem : zacina nazev vzoru. Tento vzor je pouzit na nasledujici slova
# soch-ař             #znak - oddeluje sufix od korene slova
# zub-ař#118          #komentare zacinaji znakem #
# ...                 #na slova se stale aplikuje vzor "muž"
# :bojiště            #na nasledujici slova uz vzor "bojiště"
# doskoč-iště/doskok  #pokud dochazi k alternacim, spravny vysledek je mozno uvest za znakem /
# :                   #pouze znak : znamena, ze na nasledujici slova nebude aplikovan zadny vzor
# ...
#
# ukazka vystupu:
# sochařem/soch/soch
# sochaře/soch/soch
# ...
# doskočištím/doskok/doskok
# ...

my $progname = $0;

# na vstupu ocekava jediny parametr: vstupni soubor
if (scalar @ARGV != 1) {
  print "Usage: $progname <file>\n";
  exit 1;
}
$file = shift(@ARGV); #vstupni soubor se slovy
$fileVzor = "";       #soubor s koncovkami daneho vzoru
$line = "";           #jeden radek vstupniho souboru
$kon1pad = "";        #koncovka prvniho padu daneho vzoru
$vzor = "";
@koncovky = ();
$prev = "";


open(DESCFILE, $file) or die "Nelze otevrit soubor.";

while($line = <DESCFILE>) {
  $line =~ s/\s+$//;  #odstraneni koncu radku

  if($line =~ /^:(.*?)$/)
  { #ze vstupniho souboru nacitan nazev vzoru (zacina znakem : )
    $vzor = $1; #vzor daneho slova


    if($vzor eq "") { #inicializace
      @koncovky = ("");
      next;
    }
    #otevreni souboru s koncovkami daneho vzoru
    open(FILEVZOR, "koncovky_sklonovani/slovesa/Šablona Vzor_".$vzor.".htm") or die "Nelze otevrit soubor ".$vzor;
    $line = "";
    $text = "";
    @koncovky = ();
    %hash = ();

    while($line = <FILEVZOR>) {
      $text .= $line;
    }

    close FILEVZOR;

    #koncovky vzoru jsou ulozeny v HTML tabulce
    @find = $text =~ /}}}(.*?)(\ |<)/g;

    foreach $res (@find)
    { #ziskani koncovek z tabulky
      next if $res eq " ";
      next if $res eq "<";
      next if $res eq "]]";

      if($res eq "") {
        $res = " ";
      }
      push @koncovky, $res;
      #print $res."\n"
    }

    $kon1pad = @koncovky[0];  #zapamatovani koncovky prvniho padu
    %hash = map { $_, 1 } @koncovky; #odstraneni duplicitnich koncovek
    @koncovky = keys %hash;
  }
  else
  { # ze vstupniho souboru nacitano slovo
    foreach $res (@koncovky) {
      if($res eq " ") {
        $res = "";
      }

      $slovo = $line;
      $slovo =~ s/\#.*//;     #odstraneni komentaru (zacinaji znakem #)
      $slovo =~ /(^.*?)-/;    #nalezeni pomlcky ve slove
      $zaklad = $1;           #koren slova (od zacatku slova az po pomlcku)
      $slovo =~ /.*?\/(.*)/;  #pripadny spravny vysledek oddelen od slova znakem /
      $spravnyVysledek = $1;
      $spravnyVysledek =~ s/á/a/g;  #odstraneni diakritiky ze spravneho vysledku
      $spravnyVysledek =~ s/č/c/g;
      $spravnyVysledek =~ s/ď/d/g;
      $spravnyVysledek =~ s/é/e/g;
      $spravnyVysledek =~ s/ě/e/g;
      $spravnyVysledek =~ s/í/i/g;
      $spravnyVysledek =~ s/ň/n/g;
      $spravnyVysledek =~ s/ó/o/g;
      $spravnyVysledek =~ s/ř/r/g;
      $spravnyVysledek =~ s/š/s/g;
      $spravnyVysledek =~ s/ť/t/g;
      $spravnyVysledek =~ s/ú/u/g;
      $spravnyVysledek =~ s/ů/u/g;
      $spravnyVysledek =~ s/ý/y/g;
      $spravnyVysledek =~ s/ž/z/g;

      $budizVysledek = $spravnyVysledek;
      #castecne spravny vysledek - neobsahuje koncovou samohlasku
      $budizVysledek =~ s/[aeiouy]$//;

      $slovo =~ s/-//;          #odstraneni: -
      $slovo =~ s/\/.*//;       #odstraneni: /...
      $slovo =~ s/$kon1pad$//;  #odstrani ze slova koncovky 1. padu

      if($res =~ /^[ií]/) {     #hlaskove alternace pokud koncovka zacina na i/í
        $slovo =~ s/k$/c/;
        $slovo =~ s/g$/z/;
        $slovo =~ s/ch$/š/;
        $slovo =~ s/r$/ř/;
      }
      if($res =~ /^[i]/) {
        $slovo =~ s/r$/ř/;
      }
      print  $slovo.$res."/".$spravnyVysledek."/".$budizVysledek."\n";
      $slovo = $line;
    }
  }
}

close DESCFILE;
